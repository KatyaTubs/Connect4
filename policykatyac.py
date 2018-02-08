from policies import base_policy as bp
import numpy as np
import pickle
import tensorflow as tf

BATCH_SIZE = 10
GAMMA = 0.8
EPSILON = 0.15

class Policykatyac(bp.Policy):


    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: a arg -> string value map as received in command line, notice that the "load_from" and
                            "save_to" are special arguments passing a file path that can be used for initialization
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'Morty.model.pkl'
        policy_args['load_from'] = str(policy_args['load_from']) if 'load_from' in policy_args else 'Morty.model.pkl'
        return policy_args


    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as TensorFlow sessions and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        
        # database that stores all the examples we encountered
        self.db = [] 
        # database that stores the states for current game
        self.game = []
        
        
        ########## NeuralNet Initialization ###########
        
        # The neural net consists of 3 layers, 2 of them of size input * input,
        # and the third of size input * 7, the i'th output being Q(state, i)
        
        try:
            weights = pickle.load(open(self.load_from, 'rb'))
            self.W1 = tf.Variable(weights[0], name = 'W1', dtype=tf.float32)
            self.B1 = tf.Variable(weights[1], name = 'B1', dtype=tf.float32)
            self.W2 = tf.Variable(weights[2], name = 'W2', dtype=tf.float32)
            self.B2 = tf.Variable(weights[3], name = 'B2', dtype=tf.float32)
            self.W4 = tf.Variable(weights[4], name = 'W4', dtype=tf.float32)
            self.B4 = tf.Variable(weights[5], name = 'B4', dtype=tf.float32)
            self.W3 = tf.Variable(weights[6], name = 'W3', dtype=tf.float32)
            self.B3 = tf.Variable(weights[7], name = 'B3', dtype=tf.float32)
        except:
            self.initializer = tf.contrib.layers.xavier_initializer()
            self.W1 = tf.get_variable("W1", [84, 84], dtype=tf.float32, initializer=self.initializer)
            self.B1 = tf.get_variable("B1", [84], dtype=tf.float32, initializer=self.initializer)
            self.W2 = tf.get_variable("W2", [84, 84], dtype=tf.float32, initializer=self.initializer)
            self.B2 = tf.get_variable("B2", [84], dtype=tf.float32, initializer=self.initializer)
            self.W4 = tf.get_variable("W4", [84, 84], dtype=tf.float32, initializer=self.initializer)
            self.B4 = tf.get_variable("B4", [84], dtype=tf.float32, initializer=self.initializer)
            self.W3 = tf.get_variable("W3", [84, 7], dtype=tf.float32, initializer=self.initializer)
            self.B3 = tf.get_variable("B3", [7], dtype=tf.float32, initializer=self.initializer)

        
        
        # board representation
        self.state = tf.placeholder(tf.float32, [None, 84])   
        # binary vector with i'th slot =1 iff i is legal action
        self.actions = tf.placeholder(tf.float32, [None, 7])    
        
        layer1 = tf.nn.relu(tf.matmul(self.state, self.W1) + self.B1)
        layer2 = tf.nn.relu(tf.matmul(layer1, self.W2) + self.B2)
        layer4 = tf.nn.relu(tf.matmul(layer2, self.W4) + self.B4)
        layer3 = tf.nn.relu(tf.matmul(layer4, self.W3) + self.B3)
           
        self.Q_list = tf.multiply(tf.nn.softmax(layer3), self.actions)

        self.thisQ = tf.reduce_max(self.Q_list)
        self.reward = tf.placeholder(tf.float32, [None])
        
        # placeholder for max_a'(Q(s', a')) with s' being the next state
        self.nextQ = tf.placeholder(tf.float32, [None])
        
        self.tf_gamma = tf.constant(self.gamma, tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.reward + tf.multiply(self.tf_gamma, self.nextQ) - self.thisQ))
        self.round = tf.placeholder(tf.float32, shape=())

        self.trainer = tf.train.AdamOptimizer(0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)        


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for learning and improving the policy. it accepts the
        state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        may improve the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy. use this to make your computation time smaller
                        by lowering the batch size for example...
        """
        
        
        # Don't learn from the game in which the agent lost
        if reward < 0:
            self.game = []
            return
            
        self.add_to_db(prev_state, prev_action, reward, new_state)
        
        # set positive rewards to states that let to a win
        for i in range(len(self.game)):
            self.game[i]['r'] = GAMMA**(len(self.game)-i-1) 
            self.db.append(self.game[i])

        self.game = []
        
        # sample batch of examples to learn
        sample_idx = np.random.random_integers(0, high=len(self.db)-1, size=BATCH_SIZE)
        
        actions_list = []
        nextQ_list = []
        r_list = []
        state_list = []

        for idx in sample_idx:
            actions = np.zeros(7)
            next_actions = np.zeros(7)
            actions[self.db[idx]['pre_action']] = 1
            actions_list.append(actions)
            
            next_actions[self.db[idx]['legal_moves']] = 1
            next_actions = np.reshape(next_actions, (1,7))
            
            # find the best action for the next state
            nextQ = np.max(self.sess.run(self.Q_list, {self.state : self.db[idx]['new_state'], self.actions: next_actions}))
            nextQ_list.append(nextQ)
            
            r_list.append(self.db[idx]['r'])
            state_list.append(np.reshape(self.db[idx]['prev_state'], (84)))
            
        feed_dict = {self.state: state_list,
                     self.actions: np.array(actions_list),
                     self.reward: r_list,
                     self.nextQ: nextQ_list,
                     self.round: round
                     }
        for i in range(5):
            self.sess.run(self.updateModel, feed_dict=feed_dict)

        if too_slow:
            print('learn too slow')
            
            
        
    def add_to_db(self, prev_state, prev_action, reward, new_state):
        """
        Adds the given state of the game to the game database.
        """
        
        # The board representation is two concatenated vectors - 
        # the first is the flattened board with 1 in slots with this 
        # agent's pieces
        # the second id the flattened board with -1 in slots with the other's
        # agen't pieces
        
        tmp_new_state = np.reshape(new_state, (42))
        other_id = 1 if self.id == 2 else 2

        this_new_state = np.zeros((2,42))
        this_new_state[0,tmp_new_state==self.id] = 100
        this_new_state[1,tmp_new_state==other_id] = -100
        this_new_state = np.reshape(this_new_state, (1,84))
        
        if self.mode == 'train':
        
            if prev_state is not None:
                prev_state = np.reshape(prev_state, (42))

            this_prev_state = np.zeros((2, 42))
            this_prev_state[0, prev_state==self.id] = 100
            this_prev_state[1, prev_state==other_id] = -100
            this_prev_state = np.reshape(this_prev_state, (1,84))

            self.game.append({'prev_state':this_prev_state, 'pre_action':prev_action, 'r':reward, 'new_state':this_new_state, 'legal_moves':self.get_legal_moves(new_state)})
        
        return this_new_state
        
        

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for choosing an action, given current state.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                policy. use this to make your computation time smaller
                by lowering the batch size for example...
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        
        legal_moves = self.get_legal_moves(new_state)
        new_state = self.add_to_db(prev_state, prev_action, reward, new_state)

        actions = np.zeros(7)
        actions[legal_moves] = 1
        actions = np.reshape(actions, (1,7))
        # get the Q(s,a) estimation from the model
        action_list = self.sess.run(self.Q_list, {self.state :new_state, self.actions: actions})
        #print(action_list)
        action = np.argmax(action_list)
        exploration = np.random.random()
        if too_slow:
            print('act too slow')
        
        # with decaying probability act randomly
        if self.mode == 'test' or exploration > max(self.epsilon * (self.game_duration - round)/self.game_duration, 0.05):
            return action
        
        return np.random.choice(legal_moves)
                

    def save_model(self):
        """
        A function you must implement, which returns your model, along with
        where to save your model to. The model object need to be picklable (like
        a list, for example).
        :return: a tuple: (model, save_to)
                The "model" is the policy model (e.g. for future initialization)
                The "save_to" is the path where your model will be saved. This should
                be the save_to parameter you are given from the command line
                when the game starts.
        """
        weights = []
        w1 = self.sess.run(self.W1)
        b1 = self.sess.run(self.B1)
        w2 = self.sess.run(self.W2)
        b2 = self.sess.run(self.B2)
        w3 = self.sess.run(self.W3)
        b3 = self.sess.run(self.B3)
        w4 = self.sess.run(self.W4)
        b4 = self.sess.run(self.B4)
        weights.append(w1)
        weights.append(b1)
        weights.append(w2)
        weights.append(b2)
        weights.append(w4)
        weights.append(b4)
        weights.append(w3)
        weights.append(b3)
        return weights, self.save_to
