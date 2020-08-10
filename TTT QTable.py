### TicTacToe - Self-learning with Q-Table learning
"""
Created on Fri Jul 17 19:47:48 2020
Uses Q learning to train 2 bots to play tic tac toe against each other.

@author: thomasniccum
"""
import numpy as np
import string
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler


# global constants
Blank  = 0
X_Mark = 1
O_Mark = 2


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

# dumb robotic opponent - random play
#def robot_player_v0(B, player):
#  open = B.open_squares() # get list of possible moves
#  c = np.random.randint(0,len(open))
#  print(open, open[c])
#  return open[c]

# pretty good opponent - random until it sees an opportunity to win  
def robot_player_Random(B, player):
  if player == 0:
    other_guy = 1
  else:
    other_guy = 0
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

# robot player that uses the trained QTable
def robot_player_QT(B, QTable, eps, player, debug=False):
  smoothing_eps = 0.10 # smoothes weights by this pct  
    
  if np.random.random() < eps:
     move = robot_player_Random(B, player) 
  else:
      k = B.make_key_from_board()
      actions = QTable[k][0]    # p array is allowable moves from this position
      weights = []                # z will be our list of equivalent weights to choose from
      
      for i in actions:           
          weights.append(QTable[k][1][i])
        
      maxweight = np.max(weights)
# smooth the weights in the neighborhood of max weight by an epsilon
      smoothed_weight = abs(maxweight * (1 - smoothing_eps))
      absweights = np.abs(weights)
      maxindexes = np.where(absweights >= smoothed_weight)
      if debug:
          print(maxweight, maxindexes, weights, smoothed_weight, absweights)
        
# return a move - find the highest weighted action and return the index as the action to take
# tends to cause a locked in state as a move is ALWAYS taken even if the weights are miniscully diffent
# so I propose to use an epsilon to smooth them out and create a list of possible moves that are equivalent
# and choose between them

      if len(maxindexes[0]) > 0:
          move = actions[maxindexes[0][np.random.randint(0,len(maxindexes[0]))]]
      else:
          move = actions[maxindexes[0][0]]
      if debug:
          print('Length of maxindexes: ', len(maxindexes[0]))
          print('actions: ', actions, 'Q: ', QTable[k][1], ' max:weight ', maxweight, ' move: ', move, ' maxindexes: ', maxindexes)
  return move
  
def human_player(B, Q=None, eps=0.0, player=0, debug=False):
    B.print(labels=True)
    return int(input(">> Your move, human! "))

    
def get_scaler():
      # return scikit-learn scaler object to scale the states
      # Note: you could also populate the replay buffer here
    
      states = [500, 9]
      states = np.random.randint(3, size=(500,9))
      scaler = StandardScaler()
      scaler.fit(states)
      return scaler
        

############## NN Model Class ######################
class TTT_Agent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.99 # default discount
        self.learning_rate = 0.1 # default learning rate   
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.scaler = get_scaler()
        self.randocount = 0
        self.NNcount = 0
        self.epsiloncount = 0
        self.movecount = 0
        self.bummers = []
        
#model design:
#       input layer: size of board (9)
#       hidden layers
#       output layer: size of action list (9)        

    def create(self, state_dim, action_dim, hidden_layers=1, hidden_dim=32, debug=False):
        i = Input(shape=(state_dim,))
        x = i
        for _ in range(hidden_layers):
            x = Dense(hidden_dim, activation='relu')(x)
        x = Dense(action_dim)(x)
    
        self.model = Model(i, x) 
        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])
        if debug:
            print((self.model.summary()))
        return self.model
    
    
#    def fit(game_buffer,  reward_buffer):


    def act(self, state, player): # state is the current board
        self.movecount += 1
        if np.random.rand() <= self.epsilon:
            self.epsiloncount += 1
            return robot_player_Random(state, player)
        sf = np.zeros((1,9))
        sf[0] = state.board.flatten()
#        print(">>act: state=,", sf, " with shape=", sf.shape)
        s1 = self.scaler.transform([sf[0]])
        act = self.model.predict(s1) 
        m = np.argmax(act[0])
        if state.islegal(m):
            self.NNcount += 1
            return m  # return the selected action
        else:
#                state.print()
#                print("Move: ", m)
#                input("pause::::")
                s1 = self.scaler.transform([sf[0]])
                self.update_replay_memory(sf[0], m, -1, sf[0], 0)
                self.replay()                
                self.randocount += 1
                self.bummers.append(m)
                return robot_player_Random(state, player)                
    
    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        
    def replay(self, batch_size=32):
        if self.memory.size < batch_size:
            return
        #sample a batch of data from the replay memory
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

#        print("Train on batch...")
#        print("minibatch: ", minibatch)
        
        #calculate tentative target: Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        target[done] = rewards[done]

        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target
        self.model.train_on_batch(states, target_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            

class ReplayBuffer:
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
      
  def sample_batch(self, batch_size=100):  # need to make sure we don't split an EPISODE tho unless we distribute rewards
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.current_state_buf[idxs],
                a=self.action_buf[idxs],
                r=self.reward_buf[idxs],
                s2=self.next_state_buf[idxs],
                d=self.done_buf[idxs])

def NN_play_one_game(agent, player_one_func, player_two_func, RBuf=None, is_train=False, debug=False):
      game = []         # history of moves in game buffer
      B = board()       # instantiate a board for this game 
      done = False
      states = []
      players = ['X', 'C', 'O']
      current_player = 0
      rw = [1.0, 1.0]
      gamma = 0.9
      
      while done == False:
        if debug:
          B.print(labels=True)
    
        if current_player == 0: # X to Play
          m = player_one_func(B, current_player) #do an action based on the board 
          current_state = B.board.flatten()
          B.move(players[current_player], m)
          next_state = B.board.flatten()
          s1 = agent.scaler.transform([current_state])
          s2 = agent.scaler.transform([next_state])
          states.append([s1, s2, m, 0.0, 0])
          game.append(m) # add the move to game history.
          w = B.check_done()
          if w != -1:
            done = True
            states[-1][4]
    
        else: # O to play
          m = player_two_func(B, current_player)
          current_state = B.board.flatten()
          B.move(players[current_player], m)
          next_state = B.board.flatten()
          s1 = agent.scaler.transform([current_state])
          s2 = agent.scaler.transform([next_state])
          states.append([s1, s2, m, 0.0, 0])
          game.append(m) # add the move to game history.
          w = B.check_done()
          if w != -1:
            done = True
            states[-1][4] = 1
          
      # if done - update weights with a Win or loss 
        if done:  
            if is_train:
              if w == 1:              # TIE  game
                    rw[0] = 1.0
                    rw[1] = 1.0
              else:    
                if current_player == 0: # player 0 (X) won
                    rw[0] = 1.0
                    rw[1] = 0.0
                else:                   # player 1 (O) won
                    rw[0] = 0.0
                    rw[1] = 1.0

#              if debug:  
#                  print('first update the end state with the reward')
#                  B.print()
#                  print('States: ', states,)
              ls = len(states)
              for i in range(ls):
                  #update rewardson this game
                  #add game to replay buffer 
                  sidx = -(i+1)
#                  print("Apply reward ", rw[current_player]," to state #", ls-i+1, " - ", states[-(i+1)])
                  states[sidx][3] = rw[current_player]                 
                  rw[current_player] *= gamma # discount the rewards back through the game
                  agent.update_replay_memory(states[sidx][0], states[sidx][2], states[sidx][3], states[sidx][1], states[sidx][4])
                  agent.replay()
                  if current_player == 0:
                      current_player = 1
                  else:
                      current_player = 0

        else: # not done so toggle player
          if current_player == 0:
             current_player = 1
          else:
            current_player = 0
        
      return game, w, B # return the game move history and the winner and final board postion
    
# baseN - used to convert a number to base 3 to easily construct a board vector
def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
        return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])


############## TQPModel Class ######################
class TQP_Model:
  def __init__(self):
    self.gamma = 0.99 # default discount
    self.learning_rate = 0.1 # default learning rate


    self.QTable = {} #create QTable to hold allowable actions and weights
    #initialize ATable
    for j in range(3**9):
        k = baseN(j, 3)
        k2 = k.rjust(9, '0')
        b = board()
        b.make_board_from_key(k2)
        a = b.open_squares()
        self.QTable[k2] = [a,np.zeros(9)]


  def get_actions(self, k):
    return self.QTable[k][0]
                          
  def get_weights(self, k):
    return self.QTable[k][1]

  def update_weight(self, cs, ns, action, rew=0, show=False):
    if show:
      print('Cur:',cs, 'New:', ns, 'Act:', action, 'Reward: ',rew, 'New: ',self.QTable[ns][1], 'Cur: ',self.QTable[cs][1] )
      print('Q[s,a](',self.QTable[cs][1][action], ') + lr(', self.learning_rate,') * ( rew (', rew, ') + gamma(', self.gamma,')  * max(Q(news,a*) (',
            max(self.QTable[ns][1]), ') - Q[s,a](', self.QTable[cs][1][action],')')
    self.QTable[cs][1][action] =  self.QTable[cs][1][action] + self.learning_rate * \
      (rew + self.gamma * max(self.QTable[ns][1]) - self.QTable[cs][1][action])
    if show:
      print('Updated Q[s,a]=', self.QTable[cs][1])
    return  self.QTable[cs][1][action]

  def random_move(self,k):
    possible = self.QTable[k][0]
    r = np.random.randint(0,len(possible))
    return possible[r]

  def update_game_weights(self, reward, states):
      r = reward
      i = len(states)
      i -= 2
      while i >= 0:
        r = self.update_weight(states[i][0], states[i][1], states[i][2], r, show=False)
        i -= 2


  def train_one_episode(self, epsilon_player1, epsilon_player2, debug=False):
      game = []         # history of moves in game buffer
      B = board()       # instantiate a board for this game 
      done = False
      states = []
      players = ['X', 'C', 'O']
      current_player = 0
    
      while done == False:
        if debug:
          B.print(labels=True)
    
        if current_player == 0: # X to Play
          m = robot_player_QT(B, self.QTable, epsilon_player1, current_player)
          current_state = B.make_key_from_board()
          B.move(players[current_player], m)
          next_state = B.make_key_from_board()
          states.append([current_state, next_state, m])
          game.append(m) # add the move to game history.
          w = B.check_done()
          if w != -1:
            done = True
    
        else: # O to play
          m = robot_player_QT(B, self.QTable, epsilon_player2, current_player)
          current_state = B.make_key_from_board()
          B.move(players[current_player], m)
          next_state = B.make_key_from_board()
          states.append([current_state, next_state, m])
          game.append(m) # add the move to game history.
          w = B.check_done()
          if w != -1:
            done = True
          
      # if done - update weights with a Win or loss 
        if done:  
          if current_player == 0:
              rwt = 1.0
              rww = 1.0
              rwl = 0.0
          else:
              rwt = 0.5
              rww = 1.0
              rwl = -1.0
              
          if w == 1: # we tied
            reward = rwt
          else:
            reward = rww
          if debug:  
              print('first update the end state with the reward')
              B.print()
              print('States: ', states,)
          r = self.update_weight(current_state, next_state, m, reward) 
          self.update_game_weights(r, states) # update needs to work only on last movers moves
          # now we need to update loser - his last move was his losing move...
          # pop winning/tying move off of the move list
          if debug: print("second - update losing position", w)
    
          states = states[:-1]
          if debug: print('Winning state popped off end: ', states,)
    
          # now update losing/tieing player (now the last on the list)
          cs = states[-1][0]
          ns = states[-1][1]
          move = states[-1][2]
          if w == 2:
            reward = rwt
          else:
            reward = rwl
          r = self.update_weight(cs, ns, move, reward )
          # and update losers game weights
          self.update_game_weights(r, states)
        else: # toggle player
          if current_player == 0:
             current_player = 1
          else:
            current_player = 0
        
      return game, w, B # return the game move history and the winner and final board postion



  def fit(self, epochs, verbose=False, batchsize=1000, epsilon1=.99, eps_decay1=(1 - 1e4), 
          epsilon2=.99, eps_decay2=(1 - 1e4), epsilon_min=0.01, gamma=0.99, learning_rate=0.1):    

    winners = []
    self.gamma = gamma
    self.learning_rate = learning_rate
    
    for i in range(epochs):
      game, winner, B = self.train_one_episode(epsilon1, epsilon2)
      epsilon1 = max(epsilon1 * eps_decay1, epsilon_min)
      epsilon2 = max(epsilon2 * eps_decay2, epsilon_min)
      winners.append(winner)
      if (i>1) and (i % batchsize == 0):
        xw = winners.count(0)
        cw = winners.count(1)
        ow = winners.count(2)
        tw = xw + ow + cw    
        print(game,'Games: %9d - X: % 9d % 1.3f, O: % 9d %1.3f, C: % 9d % 1.3f, eps: %1.4f' %(i, xw, xw/tw,
          ow, ow/tw, cw, cw/tw,
          epsilon1))
        winners = []

  def validate(self, validation_count, verbose=False):
    winners = []
    rats = 0
    badgames = []
    games = []
    
    print("Validating....")        
    for i in range(validation_count):
      game, B, winner = play_one_game(self, robot_player_QT, robot_player_QT, debug=False)
      if winner  != 1:
          rats += 1
          badgames.append(game)
      winners.append(winner)
      games.append(game)
      if (i>1) and (i % 1000 == 0):
        xw = winners.count(0)
        cw = winners.count(1)
        ow = winners.count(2)
        tw = xw + ow + cw    
        print('Last: ', game, 'Games: %9d - X: % 9d % 1.3f, O: % 9d %1.3f, C: % 9d % 1.3f' %(i, xw, xw/tw,
          ow, ow/tw, cw, cw/tw,
          ))
        winners = []
    
    s_games = []
    for g in games:
        s = ''
        for i in g:
            s = s + str(i)
        s_games.append(s)
    unique_games = set(s_games)

# the trained bots should ALWAYS draw - any "wins" need to be investigated! This can happen 
#   when the hyperparameters are set a little loose for extra randomness
    print("Testing generated unique game sequences: ", len(unique_games), " out of ", validation_count)
    if len(badgames) > 0:
        s_games = []
        for g in badgames:
            s = ''
            for i in g:
                s = s + str(i)
            s_games.append(s)
        unique_badgames = set(s_games)
        print("These sequences are suspicious: ", unique_badgames)

########### End TQP_Model Class ################


def play_one_game(Q, player_one_func, player_two_func, debug=False):
  game = [] # history of moves in game
  B = board()
  done = False

  
  players = ['X', 'C', 'O']
  current_player = 0

  while done == False:
    if debug:
      B.print(labels=True)

    if current_player == 0:
      m = player_one_func(B, Q.QTable, 0.0, current_player, debug=debug)
    else:
      m = player_two_func(B, Q.QTable, 0.0, current_player, debug=False)

    if not B.islegal(m):
      print("Not a legal move, try again", m)
    else:
      B.move(players[current_player], m)
      game.append(m) # add the move to game history.
      if current_player == 0:
        current_player = 1
      else:
        current_player = 0
      w = B.check_done()
      if w != -1:
        done = True
  return game, B, w # return the game move history and the winner and final board postion


##########################################################

# Train the brain
  
def main_QTable():

# instantiate the model
    Q = TQP_Model() # create the Q table object
    
# Prepare for training - set hyperparameters
    
    epsilon_player1 = 0.99              #epsilon is the probablity that a particular play will be random, high to start
    epsilon_decay_player1 = 1 - 1e-4    #epsilon_decay reduces the randomness over time as the Q Table 'learns'
    epsilon_player2 = 0.99              #split so 2 players can have different epsilons
    epsilon_decay_player2 = 1 - 1e-4
    epsilon_min = .1                    #we don't allow randomness to drop below this during training so we get more variations of the game
    learning_rate = 0.1
    gamma = 0.95
    
    Q.fit(epochs=100000, verbose=True, batchsize=10000, 
          epsilon1=epsilon_player1, eps_decay1=epsilon_decay_player1, 
          epsilon2=epsilon_player2, eps_decay2=epsilon_decay_player2, 
          epsilon_min=epsilon_min, 
          learning_rate=learning_rate,
          gamma=gamma
          )

    Q.validate(10000, verbose=False)

    print("Let's play a game...")
    while True:
        print('\n -------New Game------')
        game, B, w = play_one_game(Q, robot_player_QT, human_player, debug=False)
        B.print()
        
        if w == 1:
            print("TIE!")
        if w == 0:
            print("I WIN!")
        if w == 2:
            print("you win...")

def main_NN():
    state_dim = 9 # board size
    action_dim = 9 # possible actions
    wins = np.zeros(3)
    p = ['X', 'C', 'O']
    
# create the replay buffer
    RBuf = ReplayBuffer(state_dim, action_dim, 1000)
# instantiate the model
    model = TTT_Agent(state_dim, action_dim) # create the NN Model object
    model.create(state_dim, action_dim, hidden_layers=1, hidden_dim=32, debug=True)
    
# Training round...  play N games against self
# after each game, append S A R S' to replay buffer
# when replay buffer full, use it to train model
    for i in range(1000):
        game, w, B = NN_play_one_game(model, model.act, model.act, RBuf=RBuf, is_train=True, debug=False)
#        print("Game ", i, "Winner: ",w)
#        B.print()
        wins[w] += 1
        if ((i+1) % 100 == 0):
          print("------------")
          for i in range(3):
            print(i,":",p[i],":",wins[i], wins[i]/sum(wins), model.NNcount, model.randocount, model.epsiloncount, model.movecount)
          wins = np.zeros(3)
    print("bummers: ", len(model.bummers))
    
#    model.validate(10000, verbose=False)


# use the trained model to play a game
    print("Let's play a game...")
    while True:
        print('\n -------New Game------')
        game, w, B = NN_play_one_game(model, model.act, human_player, is_train=False, debug=False)
        B.print()
        
        if w == 1:
            print("TIE!")
        if w == 0:
            print("I WIN!")
        if w == 2:
            print("you win...")
        d = input("'nother? ")
    
    
if __name__ == '__main__':

    model_type = int(input("1 for Q Table, 2 for NN: "))
    if model_type == 1:
        main_QTable()
    else:
        main_NN()
        



