import tensorflow as tf
print(tf.__version__)
import cProfile

"""

Uses Deep Q learning to train 2 bots to play tic tac toe against each other.

@author: thomasniccum
"""
import numpy as np
from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import time 

# class board with methods for initializing, displaying, uddating the tictactoe board
# board is represented with 
#       0 = X
#       1 = a blank square
#       2 = O
# we index the Q Table with a length 9 key that is just a string version of the board status


class board:
    def __init__(self):
        self.board = np.ones((3,3), dtype=int )
        
    def print(self, labels=False):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0: print(' X ', end='')
                if self.board[i][j] == 1: 
                  if labels:
                    print ('',i*3+j,'', end='')
                  else:
                    print('   ', end='')
                if self.board[i][j] == 2: print(' O ', end='')
                if j < 2: print('|', end='')
            print('')
            if i < 2: print('-----------')
        print('')

# given a length 9 string key such as '010020010', will construct a board
    def make_board_from_key(self, k):
      for m in range(9):
        i = m // 3
        j = m - (i*3)
        self.board[i][j] = int(k[m])
 
# given a board, will construct a length 9 key - used as a 'state' in the QTable for determining next action
    def make_key_from_board(self):
      m = ''
      for i in range(3):
        for j in range(3):
          m = m + str(self.board[i][j])
      return m

# open_squares - returns the available blank squares on the board
    def open_squares(self):
        return np.where(self.board.flatten() == 1)[0]

# updates the board with a players move
    def move(self, player, m):
        i = m // 3
        j = m - (i*3)
        if player == 'X':
            k = 0
        else:
            k = 2
        self.board[i][j] = k

# islegal returns true if a proposed move is legal/false otherwise
    def islegal(self, m):
        i = m // 3
        j = m - (i*3)
        if i < 0 or i > 2: 
          r = False
        else:
          if j < 0 or j > 2: 
            r = False
          else:
            r = self.board[i][j] == 1
        return r

# check for game over.. returns winning player, X=0, O=1, tie=2
    def check_done(self):
        winner = -1 # not done
        #check rows: 
        for i in range(3):
          if (self.board[i,0] == self.board[i,1]) and (self.board[i,1] == self.board[i,2]) and (self.board[i,0] != 1):
            winner = self.board[i,0]
            break

        #check cols: 
        if winner == -1:
          for j in range(3):
            if (self.board[0,j] == self.board[1,j]) and (self.board[1,j] == self.board[2,j]) and (self.board[0,j] != 1):
              winner = self.board[0,j]
              break

        #check diagonals:       
        if winner == -1:
          if (self.board[0,0] == self.board[1,1]) and (self.board[1,1] == self.board[2,2])  and (self.board[0,0] != 1):
            winner = self.board[0,0]
          else:
            if (self.board[0,2] == self.board[1,1]) and (self.board[1,1] == self.board[2,0])  and (self.board[2,0] != 1):
              winner = self.board[0,2] 
          
        if winner == -1:
          tie = True
          if len(self.open_squares() > 0):
            tie = False
          if tie:
            winner = 1    # tie game, no more legal moves
        return winner # 0=X, 1=Tie, 2=O -1 = not done
  
#  quick and dirty "strategy" for robotic player imporovement...
#    simply checks to see if:
#       A.  If we can win by moving to a square
#       B.  If we can't win do we need to block the other guy...
#       returns a list of suggested moves

    def check_need_one_to_win(self, player):
        winners = []
        if player == 0: 
          p = 0
        else:
          p = 2
        #check rows: 
        for i in range(3):
          if (self.board[i][0] == self.board[i][1]) and (self.board[i][1] == p) and (self.board[i][2] == 1):
            winners.append(i*3+2)
          if (self.board[i][1] == self.board[i][2]) and (self.board[i][1] == p) and (self.board[i][0] == 1):
            winners.append(i*3+0)
          if (self.board[i][0] == self.board[i][2]) and (self.board[i][2] == p) and (self.board[i][1] == 1):
            winners.append(i*3+1)
        #check cols: 
        for j in range(3):
          if (self.board[0][j] == self.board[1][j]) and (self.board[1][j] == p) and (self.board[2][j] == 1):
            winners.append(2*3+j)
          if (self.board[1][j] == self.board[2][j]) and (self.board[1][j] == p) and (self.board[0][j] == 1):
            winners.append(0*3+j)
          if (self.board[0][j] == self.board[2][j]) and (self.board[2][j] == p) and (self.board[1][j] == 1):
            winners.append(1*3+j)

        #check diagonals:       
        if (self.board[0][0] == self.board[1][1]) and (self.board[1][1] == p) and (self.board[2][2] == 1):
            winners.append(8)
        if (self.board[0][0] == self.board[2][2]) and (self.board[2][2] == p) and (self.board[1][1] == 1):
            winners.append(4)
        if (self.board[1][1] == self.board[2][2]) and (self.board[2][2] == p) and (self.board[0][0] == 1):
            winners.append(0)
        if (self.board[0][2] == self.board[1][1])  and (self.board[1][1] == p) and (self.board[2][0] == 1):
            winners.append(6)
        if (self.board[0][2] == self.board[2][0])  and (self.board[2][0] == p) and (self.board[1][1] == 1):
            winners.append(4)
        if (self.board[1][1] == self.board[2][0])  and (self.board[1][1] == p) and (self.board[0][2] == 1):
            winners.append(2)

        return list(set(winners))

# PLAYER FUNCTIONS
# - robot_player_Random - checks to see if it can win on this more, or needs to block, otherwise random
# - human_player - shows the board and asks for keyboard input move

def robot_player_learner(B, player):
#  if player == 0:
#    other_guy = 1
#  else:
#    other_guy = 0

#  open = B.check_need_one_to_win(player) # get list of possible moves
# do I have a winning move - take it
# if len(open) > 0:
 #   c = open[0]
 # else:
# does the OTHER guy have a winning move - block it    
 #   open = B.check_need_one_to_win(other_guy) # get list of possible moves
 #   if len(open) > 0:
 #     c = open[0]
 #   else:
 #any open square
      open = B.open_squares() # get list of possible moves
      r = np.random.randint(0,len(open))
      c = open[r]
      return c

def robot_player_teacher(B, player):
  corners = [0,2,6,8]
  if player == 0:
    other_guy = 1
  else:
    other_guy = 0
  first_move = np.where(B.board.flatten() != 1)
  if len(first_move[0]) == 1:  # we are making our first move
      if first_move[0][0] == 4:  # he chose center, we'll take any corner
          corner = np.random.randint(0,4, dtype=int)
          return corners[corner]
      if first_move[0][0] in corners:
          return 4 # choose the center

  open = B.check_need_one_to_win(player) # get list of possible moves
# do I have a winning move - take it
  if len(open) > 0:
    c = open[0]
  else:
# does the OTHER guy have a winning move - block it    
    open = B.check_need_one_to_win(other_guy) # get list of possible moves
    if len(open) > 0:
      c = open[0]
    else:
 #any open square
      open = B.open_squares() # get list of possible moves
      r = np.random.randint(0,len(open))
      c = open[r]
  return c

 
def human_player(B, Q=None, eps=0.0, player=0, debug=False):
    legal = False
    B.print(labels=True)
    while not legal:
      m = int(input(">> Your move, human! "))
      legal = B.islegal(m)
      if not legal:
        print("Ooops... try a LEGAL move this time!")
    return m  
    
      
# return a move - find the highest weighted action and return the index as the action to take
# tends to cause a locked in state as a move is ALWAYS taken even if the weights are miniscully diffent
# so I propose to use an epsilon to smooth them out and create a list of possible moves that are equivalent
# and choose between them
def get_best_legal_move(state, output, smoothing_eps, debug=False):
      weights = []                # z will be our list of equivalent weights to choose from
      actions = []
      for i in range(9):           
          if state.islegal(i):
              weights.append(output[i])
              actions.append(i)

#find the highest weighted move
      maxweight = np.max(weights)
# smooth the weights in the neighborhood of max weight by an epsilon
      if maxweight >= 0:
          smoothed_weight = maxweight * (1 - smoothing_eps)
      else:
          smoothed_weight = maxweight * (1 + smoothing_eps)
          
      maxindexes = np.where(weights >= smoothed_weight)
      if (len(maxindexes[0]) == 0):
        print("Learner Smoothing error: ", weights, actions, maxindexes[0])

      move = actions[maxindexes[0][np.random.randint(0,len(maxindexes[0]))]]
      if debug:
          print("Best move: ", state.board.flatten(), move, maxindexes[0], weights, actions, len(maxindexes[0]))
      return move
       
def get_best_legal_move_with_checking(state, output, smoothing_eps, debug=False):
  player = 1
  other_guy = 0
  open = state.check_need_one_to_win(player) # get list of possible moves
# do I have a winning move - take it
  if len(open) > 0:
    return open[0]
  else:
# does the OTHER guy have a winning move - block it    
    open = state.check_need_one_to_win(other_guy) # get list of possible moves
    if len(open) > 0:
      return open[0]
    else:      
        weights = []                # z will be our list of equivalent weights to choose from
        actions = []
        for i in range(9):           
          if state.islegal(i):
              weights.append(output[i])
              actions.append(i)

#find the highest weighted move
        maxweight = np.max(weights)
# smooth the weights in the neighborhood of max weight by an epsilon
        if maxweight >= 0:
          smoothed_weight = maxweight * (1 - smoothing_eps)
        else:
          smoothed_weight = maxweight * (1 + smoothing_eps)
        maxindexes = np.where(weights >= smoothed_weight)
        if (len(maxindexes[0]) == 0):
          print("Trainer Smoothing Error: ", weights, actions, maxindexes[0])
# return a move - find the highest weighted action and return the index as the action to take
# tends to cause a locked in state as a move is ALWAYS taken even if the weights are miniscully diffent
# so I propose to use an epsilon to smooth them out and create a list of possible moves that are equivalent
# and choose between them

        move = actions[maxindexes[0][np.random.randint(0,len(maxindexes[0]))]]
        if debug:
              print("Best move: ", state.board.flatten(), move, maxindexes[0], weights, actions, len(maxindexes[0]))
        return move
       

############## NN Model Class ######################
class TTT_Agent():
    def __init__(self, state_size, action_size, load=False, filename=''):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = replay_buffer(state_size, action_size, size=500)
#        self.gamma = 0.90 # default discount
        self.learning_rate = 0.05 # default learning rate   
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.best_move_smoothing = 0.1 # when several next best moves are roughly equal, choose randomly
        self.randocount = 0
        self.NNcount = 0
        self.epsiloncount = 0
        self.movecount = 0
        self.bummers = []
        if load == True:
          self.model = load_model(filename)
#model design:
#       input layer: size of board (9)
#       hidden layers
#       output layer: size of action list (9)        

    def create(self, state_dim, action_dim, hidden_layers=1, hidden_dim=(32), debug=False):
        i = Input(shape=(state_dim,))
        x = i
        for k in range(hidden_layers):
            x = Dense(hidden_dim[k], activation='relu')(x)
        x = Dense(action_dim)(x) # output layer
        self.model = Model(i, x) 
        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])
        if debug:
            print((self.model.summary()))
        return self.model
    
    def model_save(self, filename):
      self.model.save(filename)

    def pred(self, state):
        return self.model.predict(state)

    def act_learner(self, state, player): # state is the current board
        self.movecount += 1
        if np.random.rand() <= self.epsilon:
            self.epsiloncount += 1
            return robot_player_learner(state, player)
        sf = np.zeros((1,9))
        sf[0] = state.board.flatten()

        act = self.model.predict(sf) 
 
        m = get_best_legal_move(state, act[0], self.best_move_smoothing, debug=False)
        self.NNcount += 1
        return m            
    
    def act_trainer(self, state, player): # state is the current board
        self.movecount += 1
        if np.random.rand() <= self.epsilon:
            self.epsiloncount += 1
            return robot_player_teacher(state, player)
        sf = np.zeros((1,9))
        sf[0] = state.board.flatten()

        act = self.model.predict(sf) 
 
        m = get_best_legal_move_with_checking(state, act[0], self.best_move_smoothing, debug=False)
        self.NNcount += 1
        return m            
    
    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        
    def printmini2(self, mini, t1, t2, t3, t4):
        print("\n")
        print("         current states      a    r              next state        d   tgt             predict(states) pre train                                       Target_full                                                      predict(states) post train                      ")
        print("---------------------------- - ------ ---------------------------- - ------ -------------------------------------------------------------- -------------------------------------------------------------- --------------------------------------------------------------")
        for i in range(len(mini["s"])):
            print("%20s %1d %+1.3f %20s %1d %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f %+1.3f" % \
                  (mini["s"][i], mini["a"][i], mini["r"][i], mini["s2"][i], mini["d"][i], \
                   t1[i], \
                   t2[i][0], t2[i][1], t2[i][2], t2[i][3], t2[i][4], \
                       t2[i][5], t2[i][6], t2[i][7], t2[i][8],       \
                   t3[i][0], t3[i][1], t3[i][2], t3[i][3], t3[i][4], \
                       t3[i][5], t3[i][6], t3[i][7], t3[i][8],       \
                   t4[i][0], t4[i][1], t4[i][2], t4[i][3], t4[i][4], \
                       t4[i][5], t4[i][6], t4[i][7], t4[i][8]        \
                           ))
        print("-----------------------------------\n")
            

    def replay_one_game(self, batch_size=32, debug=False):
        #sample a batch of data from the replay memory
        if self.memory.size < batch_size:
            return
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']
# chanaging the rewards application to happen at end of each game - in the play one game function
        target = rewards # + (1 - done) * self.gamma * future_rwds
        target[done] = rewards[done] # if there are any "done" then replace the target with the reward

        target_full = self.model.predict(states)
        predict_pre = self.model.predict(states)
        target_full[np.arange(len(actions)), actions] = target
        
        self.model.train_on_batch(states, target_full)

        if debug == True:
         predict_post = self.model.predict(states)
         self.printmini2(minibatch, target, predict_pre, target_full, predict_post)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
            

class replay_buffer:
  def __init__(self, obs_dim, act_dim, size):
    self.current_state_buf = np.zeros([size, obs_dim], dtype = np.float32)  # board state pre move
    self.next_state_buf = np.zeros([size, obs_dim], dtype = np.float32)  # board state post move
    self.action_buf = np.zeros(size, dtype = np.uint8)  # board state pre move
    self.reward_buf = np.zeros(size, dtype = np.float32)  # rewards
    self.done_buf = np.zeros(size, dtype = np.uint8)  # board state pre move
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, current_state, action, reward, next_state, done):
    self.current_state_buf[self.ptr] = current_state
    self.next_state_buf[self.ptr] = next_state
    self.action_buf[self.ptr] = action
    self.reward_buf[self.ptr] = reward
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)
#    print("Store: ptr=",self.ptr, " size:",self.size)
      
  def sample_batch(self, batch_size=32):  # need to make sure we don't split an EPISODE tho unless we distribute rewards
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.current_state_buf[idxs],
                a=self.action_buf[idxs],
                r=self.reward_buf[idxs],
                s2=self.next_state_buf[idxs],
                d=self.done_buf[idxs])

  def game_batch(self):  # return everything in the buffer - should be one game's worth
        d = dict(s=self.current_state_buf[0:self.ptr],
                a=self.action_buf[0:self.ptr],
                r=self.reward_buf[0:self.ptr],
                s2=self.next_state_buf[0:self.ptr],
                d=self.done_buf[0:self.ptr])
        self.ptr = 0
        return d

def NN_play_one_game(agent, player_one_func, player_two_func, RBuf=None, is_train=False, debug=False):
      game = []         # history of moves in game buffer
      B = board()       # instantiate a board for this game 
      done = False
      states = []
      players = ['X', 'C', 'O']
      current_player = 0
      rw = [1.0, 1.0]
      gamma = 0.6
      
      if is_train == False:
          agent.epsilon = 0.0 # no random moves if not training
          agent.best_move_smoothing = 0.0 # no smoothing if not training
          
      while done == False:
        if debug:
          B.print(labels=True)
    
        if current_player == 0: # X to Play
          m = player_one_func(B, current_player) #do an action based on the board 
          current_state = B.board.flatten()
          B.move(players[current_player], m)
          next_state = B.board.flatten()
#          s1 = agent.scaler.transform([current_state])
#          s2 = agent.scaler.transform([next_state])
          states.append([current_state, next_state, m, 0.0, 0])
          game.append(m) # add the move to game history.
          w = B.check_done()
          if w != -1:
            done = True
            states[-1][4] = 1 # set DONE for the winner
            states[-2][4] = 1 # set DONE for the loser     
        else: # O to play
          m = player_two_func(B, current_player)
          current_state = B.board.flatten()
          B.move(players[current_player], m)
          next_state = B.board.flatten()
#          s1 = agent.scaler.transform([current_state])
#          s2 = agent.scaler.transform([next_state])
          states.append([current_state, next_state, m, 0.0, 0])
          game.append(m) # add the move to game history.
          w = B.check_done()
          if w != -1:
            done = True
            states[-1][4] = 1 # for the winner
            states[-2][4] = 1 # for the loser 
          
      # if done - update weights with a Win or loss 
        if done and is_train:  
            if is_train:
              if w == 1:              # TIE  game
                    rw[0] = 1.0
                    rw[1] = 1.0
              else:    
                if current_player == 0: # player 0 (X) won
                    rw[0] = 1.0
                    rw[1] = -1.0
                else:                   # player 1 (O) won
                    rw[0] = -1.0
                    rw[1] = 1.0

              ls = len(states)
              for i in range(ls):
                  #update rewardson this game
                  #add game to replay buffer 
                  sidx = -(i+1)
                  states[sidx][3] = rw[current_player]                 
                  rw[current_player] *= gamma  # discount the rewards back through the game
                  agent.update_replay_memory(states[sidx][0], states[sidx][2], states[sidx][3], states[sidx][1], states[sidx][4])

                  if current_player == 0:
                      current_player = 1
                  else:
                      current_player = 0
              agent.replay_one_game(debug=debug)
        else: # not done so toggle player
          if current_player == 0:
             current_player = 1
          else:
            current_player = 0
        
      return game, w, B # return the game move history and the winner and final board postion
    
# baseN - used to convert a number to base 3 to easily construct a board vector
def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
        return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])


##########################################################

# Train the brain
def train_NN(pass_size=100):
    debug = False
    state_dim = 9 # board size
    action_dim = 9 # possible actions
    wins = np.zeros(3)
    p = ['X', 'C', 'O']
    games = []
    
# create the replay buffer
    RBuf = replay_buffer(state_dim, action_dim, 1000)
    model = TTT_Agent(state_dim, action_dim) # create the NN Model object
# instantiate the model
    print("create agent")
    tl = ''
    while tl != 'T' and tl != 'L' and tl != 'B':
      tl = input('Train New (T) or Load Existing (L) model or Load Existing and Train More (B): ')
    if (tl == 'T') or (tl == 'B'):
      g = int(input("Number of training games to play: "))
      print("Training will take about", g*.75/60, "minutes.")
    if tl == 'B' or tl == 'L':
      print("create & load model old model")
      model = TTT_Agent(state_dim, action_dim, load=True, filename='NN TTT.h5') # create the NN Model object
      model.epsilon = 0.0
    else: #(tl == 'T')
  # Training round...  play N games against self
  # after each game, append S A R S' to replay buffer
  # when replay buffer full, use it to train model
      model.create(state_dim, action_dim, hidden_layers=2, hidden_dim=[36, 36], debug=debug)

    if (tl == 'B') or (tl == 'T'):
      training_start_time = time.time()
      training_pass_start_time = time.time()
      for j in range(g):
          game, w, B = NN_play_one_game(model, model.act_learner, model.act_trainer, RBuf=RBuf, is_train=True, debug=debug)
          games.append(game)
          wins[w] += 1
          if ((j+1) % pass_size == 0):
            print(" games player wins pct     NNs random epsilon moves")
            for i in range(3):
              print("%6d: %c - %6d %1.2f %6d %6d %6d %6d" % \
                  (j+1,p[i],wins[i], wins[i]/sum(wins), \
                      model.NNcount, model.randocount, model.epsiloncount, model.movecount))
            print("------------")
            print("Game: ", game)
            print("------------")
            wins = np.zeros(3)
            model.NNcount = 0 
            model.randocount = 0 
            model.epsiloncount = 0 
            model.movecount = 0
            model.model_save('NN TTT.h5') # save a copy of current state - then we can break out
            training_pass_stop_time = time.time()
            total_pass_training_time =  training_pass_stop_time - training_pass_start_time
            print("This pass's training took ",total_pass_training_time, " seconds, or ", total_pass_training_time / pass_size, " seconds per game.")
            print("Estimated time remaining: ", (g - j)*(total_pass_training_time / pass_size)/60, "minutes.")
            print("epsilon: ", model.epsilon)
            training_pass_start_time = time.time()
      print("Saving model...")
      model.model_save('NN TTT.h5')
      training_stop_time = time.time()
      total_training_time =  training_stop_time - training_start_time
      print("training took ",total_training_time, " seconds, or ", total_training_time / g, " seconds per game.")
    return model, games

# use the trained model to play a game
def play_a_game(model):
    print("Let's play a game...")
    d = ''
    while True and d != 'n':
        print('\n -------New Game------')
        game, w, B = NN_play_one_game(model, model.act_learner, human_player, is_train=False, debug=False)
        B.print()
        
        if w == 1:
            print("TIE!")
        if w == 0:
            print("I WIN!")
        if w == 2:
            print("you win...")
        d = input("'nother? (n to quit) ")
     
def test_a_board():
            model.epsilon = 0.0
            b = list(input("enter a board:"))
            B = board()
            B.make_board_from_key(b)
            B.print()
            sf = np.zeros((1,9))
            sf[0] = B.board.flatten()
            a = model.pred(sf)
            print("Model Prediction: ", a) 
            a2 = model.act_learner(B, 0)
            print("Action: ", a2)

def game_search(games):
    from collections import Counter
    gamepad = []
    
    for i in range(len(games)):
        s = ''
        for j in range(9):
            if j < len(games[i]):
                s = s + str(games[i][j])
            else:
                s = s + '0'
        gamepad.append(s)        
    gd = Counter(gamepad)
    gl = sorted(gd.items(), reverse=True, key=lambda x: x[1])
    print("Game Frequency: ", len(gl))
    for gk in gl:
      if gk[1] > 1:  
        print(gk[0], "::", gk[1])



if __name__ == '__main__':


#        model, games = train_NN() # return the model for interactive work
        while True:
            a = ""
            while a not in ["T", "P", "B", "S", "Q"]:
              a = input("Train (T), Play (P), Board (B), Search Games (S) or Quit (Q):")
            if a == "T":
                model, games = train_NN()
            if a == "P":
                play_a_game(model)
            if a == "B":
                test_a_board()
            if a == "S":
                game_search(games)
            if a == "Q":
                break