# BallRadar: Ball Trajectory Inference from Player Movements in Football Matches

### Abstract
As artificial intelligence spreads out to numerous fields, the application of AI to sports analytics is also in the spotlight. However, one of the major challenges is the difficulty of automated acquisition of the continuous movement data during sports matches. In particular, it is a conundrum to reliably track a tiny ball on a wide soccer pitch with obstacles such as occlusion and imitations. Tackling this problem, this paper proposes an inference framework of ball trajectory from player trajectories as a cost-efficient alternative to ball tracking. Given a sequence of (x,y)-coordinates of players, our neural network first estimates the "rough" ball trajectory. We employ the Set Transformer architecture to effectively embed the set of player trajectories in a permutation-invariant manner and the original Transformer for estimating a rough ball trajectory from the Set Transformer embedding. Then, we post-process the rough trajectory to restore a more "natural" trajectory by rule-based estimation of the ball possession status (in possession by a player or transition between players).

### Data
Download link : http://gofile.me/6UFJt/0NZvyRzXx
