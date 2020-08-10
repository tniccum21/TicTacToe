# TicTacToe
TicTacToe with Reinforcement Learning

DeepQ version uses a true NN
QTable version uses a Qtable rather than a NN

DeepQ version:

It knows nothing about the game except what a legal move is.  It learns by playing games against itself.

At the beginning you select T for train and give it a number of training games to run.  If you choose T for training:
you then can Train a new model, Load an existing model, or Load existing and Train it some more.

Two 'bots play against each other and both learn.  The bot playing O has a bit of knowledge built in (it checks to see if it can win on this move or if it needs to block on this move to prevent an X win).  I added that to enhance the training of the X bot.  After training is done, the model weights are saved and can be reloaded if you wish... more training can be done or you can play vs. the X bot.  

After training, you can use the S command to dump the list of games it played through.  I used this in testing as I found that without  a lot of randomness added, the games would quickly converge and the bots would get locked into playing the same game over and over again.  It's still that way, but really there's only so many ways to play TTT.

The training loop basically plays a game, and assigns rewards based on outcome.  The rewards are propagated back through the moves leading to the outcome, but discounted for each move back in history.  The game is then inserted into a buffer.  Games are added to the buffer until we reach a certain batch size, then the buffer is randomly sampled and the sample is used for training the neural network model.  In this way the play starts randomly until the network training reaches a certain point and the NN takes over.  This happens gradually  using a parameter called epsilon - using the epsilon-greedy method... play is divided between random and NN driven based on the value of epsilon, which decays over time to a small number (0.01) so 1% - 100% of play is  random during training to help explore the game space and not get too locked in.

