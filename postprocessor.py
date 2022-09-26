import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib import animation


class Postprocessor:
    def __init__(self, input_traces, episode, target_traces=None,
                 pred_traces=None, pred_team_poss=None, target_type='ball'):
        self.episode = episode

        self.input_traces = input_traces
        self.target_traces = target_traces
        self.pred_team_poss = pred_team_poss
        self.pred_traces = pred_traces
        
        self.num_players = self.input_traces.shape[1] // 4
        self.trans_periods = None
        self.accel_thres = 0.3
        self.thres_dist = 1
        self.thres_cos = 0.7
        
        self.target_type = target_type
        self.ball_df = None
        if self.target_type == 'ball' and self.pred_traces is not None:
            self.ball_df = pd.DataFrame(self.pred_traces, columns=['x', 'y'])
        
    @staticmethod
    def rolling(a, n=5):
        # n must be an odd integer
        a = np.pad(a, (n // 2, n // 2), mode='edge')
        cumsum = np.cumsum(a, dtype=float)
        cumsum[n:] = cumsum[n:] - cumsum[:-n]
        return cumsum[n - 1:] / n
        
    @staticmethod
    def detect_transitions(ball_df, accel_thres, episode_num):
        accel_df = ball_df[['accel']].copy()
        touch_df = ball_df[['touch']].copy()
        
        for k in np.arange(2) + 1:
            accel_df[f'prev{k}'] = accel_df['accel'].shift(k, fill_value=0)
            accel_df[f'next{k}'] = accel_df['accel'].shift(-k, fill_value=0)

        max_flags = ((accel_df['accel'] == accel_df.max(axis=1)) & (accel_df['accel'] > accel_thres)) #| (touch_df['touch'] > 0)
        min_flags = (accel_df['accel'] == accel_df.min(axis=1)) & (accel_df['accel'] < accel_thres * -1)
        max_times = accel_df[max_flags].index.tolist()
        min_times = accel_df[min_flags].index.tolist()

        # Following part is to ensure that acceleration should come first and decceleration should come last
        trans_periods = []
        try:
            if max_times[0] > min_times[0]: # if acceleration is later than decceleration
                min_times.pop(0)
            if max_times[-1] > min_times[-1]:  # if final acceleration is later than final decceleration
                min_times.append(accel_df.shape[0])
            
            for i in range(len(max_times)):
                t_max = max_times[i]
                while t_max > min_times[0]:
                    min_times.pop(0)
                
                t_min_candidates = []
                if i+1 < min(len(max_times), len(min_times)):
                    while max_times[i+1] > min_times[0]:
                        t_min_candidates.append(min_times.pop(0) - 1)
                else:
                    while len(max_times) < len(min_times):
                        t_min_candidates.append(min_times.pop(0) - 1)
                t_min_candidates.append(min_times[0] - 1)
                t_min = ball_df.loc[t_min_candidates, 'speed'].idxmin()
                
                if i+1 < len(max_times) and t_min > max_times[i+1]:
                    min_times.insert(i, t_max)
                else:
                    trans_periods.append((t_max, t_min + 1))
        except IndexError as e:
            print(f'{e} in episode {episode_num}')
            pass
        
        return trans_periods
    
    @staticmethod
    def remove_poss_spikes(ball_df):
        if ball_df['poss_id'].iloc[0] != 0:
            # scores = -(ball_df['speed'] + 1) / (ball_df['nearest_dist'] + 1)
            # player_poss = ball_df.at[scores.idxmax(), 'player_poss']
            player_poss = ball_df['nearest'].value_counts().index[0]
            ball_df.loc[ball_df.index, 'player_poss'] = player_poss
        return ball_df
        
    def construct_ball_df(self):
        player_x = self.input_traces[:, 0::4]
        player_y = self.input_traces[:, 1::4]
        player_xy_stack = np.stack([player_x, player_y], axis=2)
        pred_xy_stack = self.pred_traces[:, np.newaxis, :] # seq_len x 1 x 2

        ball_dists = np.linalg.norm(player_xy_stack - pred_xy_stack, axis=2) # distance between predicted ball and each players.
        
        if self.pred_team_poss is not None:
            team1_w = (self.pred_team_poss[:, 2] / (self.pred_team_poss[:, 1] + 1e-5))[:, np.newaxis]
            team1_w = 1
            team2_w = 1 / (team1_w)
            
            team1_penalty = np.concatenate([np.zeros((self.pred_team_poss.shape[0], 11)), np.ones((self.pred_team_poss.shape[0], 11))], -1) * self.pred_team_poss[:, 1, np.newaxis] * 100
            team2_penalty = np.concatenate([np.ones((self.pred_team_poss.shape[0], 11)), np.zeros((self.pred_team_poss.shape[0], 11))], -1) * self.pred_team_poss[:, 2, np.newaxis] * 100
            
            ball_dists_weighted = np.concatenate([
                ball_dists[:, :self.num_players//2] * team1_w,
                ball_dists[:, self.num_players//2:] * team2_w
            ], axis=1) + team1_penalty + team2_penalty
            
            self.ball_df['nearest'] = (np.argmin(ball_dists_weighted, axis=1)).astype(int) + 1
            
            # print(ball_dists_weighted.shape)
        else:
            self.ball_df['nearest'] = (np.argmin(ball_dists, axis=1)).astype(int) + 1
            
        self.ball_df['nearest_dist'] = ball_dists[(np.arange(self.ball_df.shape[0]), self.ball_df['nearest'] - 1)]

        vels = (self.ball_df[['x', 'y']].diff() / 0.1)
        self.ball_df['speed'] = vels.apply(np.linalg.norm, axis=1)
        self.ball_df['accel'] = self.ball_df['speed'].diff().shift(-1).fillna(0)
        self.ball_df['speed'] = self.ball_df['speed'].fillna(0)

        self.ball_df['poss_id'] = 0
        self.ball_df['team_poss'] = 0
        if self.pred_team_poss is not None:
            self.ball_df['team_poss'] = self.pred_team_poss.argmax(axis=-1)
        self.ball_df['player_poss'] = self.ball_df['nearest']
        
        self.ball_df[['vel_x', 'vel_y']] = self.ball_df[['x', 'y']].diff().fillna(0) / 0.1
        vels = self.ball_df[['vel_x', 'vel_y']]
        self.ball_df['speed'] = vels.apply(np.linalg.norm, axis=1)
        speeds = self.ball_df['speed']
        self.ball_df['speed_smooth'] = self.rolling(speeds.values)
        vels_next = vels.shift(-1)
        speeds_next = speeds.shift(-1)
        
        eps=1e-10
        
        cos_num = np.sum(vels * vels_next, axis=1) + eps
        cos_denom = speeds * speeds_next + eps
        self.ball_df['cos'] = cos_num / cos_denom
        
        
        # Consider that a "touch" occurs when the ball is close enough to a player
        # or the direction of the ball is significantly changed
        self.ball_df['touch'] = np.where(
            (self.ball_df['cos'] < self.thres_cos) |
            (self.ball_df['nearest_dist'] < self.thres_dist),
            self.ball_df['nearest'], 0
        )

        # trans_starts = self.ball_df[(self.ball_df['accel'] > 0.7).astype(int).diff() == 1].index.tolist()
        # trans_ends = (self.ball_df[
        #     ((self.ball_df['accel'] < -0.7) & (self.ball_df['speed'] < 12)).astype(int).diff() == 1
        # ].index - 1).tolist()

        # if trans_starts[0] > trans_ends[0]:
        #     trans_starts.insert(0, 0)

        # for i, t_from in enumerate(trans_starts):
        #     t_next = trans_starts[i+1] if i < len(trans_starts) - 1 else self.ball_df.shape[0]

        #     if (len(trans_ends) <= i) or (trans_ends[i] > t_next):
        #         t_to = self.ball_df['accel'][t_from:t_next].idxmin() - 1
        #         trans_ends.insert(i, t_to)
        #     else:
        #         while trans_ends[i] < t_from:
        #             trans_ends.remove(trans_ends[i])
        #         t_to = trans_ends[i]

        self.trans_periods = self.detect_transitions(self.ball_df, self.accel_thres, self.episode)
        for t_from, t_to in self.trans_periods:
            self.ball_df.loc[t_from:t_to, 'player_poss'] = 0

        poss_prev = self.ball_df['player_poss'].shift(1, fill_value=0)
        self.ball_df['poss_id'] = ((poss_prev == 0) & (self.ball_df['player_poss'] != 0)).astype(int).cumsum()
        self.ball_df['poss_id'] = np.where(self.ball_df['player_poss'] != 0, self.ball_df['poss_id'], 0)
        # self.ball_df = self.ball_df.groupby('poss_id').apply(self.remove_poss_spikes)

        self.ball_df['x_revise'] = np.nan
        self.ball_df['y_revise'] = np.nan

        poss_idxs = self.ball_df[self.ball_df['player_poss'] > 0].index
        player_poss = self.ball_df.loc[poss_idxs, 'player_poss'].values
        self.ball_df.loc[poss_idxs, ['x_revise', 'y_revise']] = player_xy_stack[(poss_idxs, player_poss - 1)]
        self.ball_df = self.ball_df.interpolate()
        
    def visualize(self, feats, pitch_size=(108, 72), max_frames=200, true_gk_traces=None):
        team1_x = self.input_traces[:, 0:self.num_players * 2:4]
        team1_y = self.input_traces[:, 1:self.num_players *2:4]
        team2_x = self.input_traces[:, self.num_players * 2::4]
        team2_y = self.input_traces[:, self.num_players * 2+1::4]
        
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        ax.axis([0, pitch_size[0], 0, pitch_size[1]])

        if self.pred_team_poss is not None:
            team1_size = 400 + 400 * self.pred_team_poss[0, 0]
            team2_size = 400 + 400 * self.pred_team_poss[0, 1]
        else:
            team1_size = 600
            team2_size = 600

        team1_scat = ax.scatter(team1_x[0], team1_y[0], s=team1_size, c='tab:red')
        team2_scat = ax.scatter(team2_x[0], team2_y[0], s=team2_size, c='tab:blue')

        player_lines = dict()
        player_annots = dict()
        for i in range(self.num_players):
            if i < self.num_players // 2:
                color = 'tab:red'
                player_id = i + 1
            else:
                color = 'tab:blue'
                player_id = i + 1 if self.target_type == 'ball' else i + 2
            
            player_lines[i], = ax.plot([], [], color=color)
            player_annots[i] = ax.annotate(
                player_id, xy=self.input_traces[0, 4*i:4*i+2], ha='center', va='center',
                color='w', fontsize=15, fontweight='bold', annotation_clip=False
            )
            player_annots[i].set_animated(True)

        if self.target_type == 'ball':
            if self.target_traces is not None:
                target_x = self.target_traces[:, 0]
                target_y = self.target_traces[:, 1]

                target_scat = ax.scatter(target_x[0], target_y[0], s=150, facecolors='w', edgecolors='k')
                target_line, = ax.plot([], [], 'k')

            if self.pred_traces is not None:
                pred_x = self.ball_df['x']
                pred_y = self.ball_df['y']
                pred_x_revise = self.ball_df['x_revise']
                pred_y_revise = self.ball_df['y_revise']

                pred_scat = ax.scatter(pred_x[0], pred_y[0], s=300, c='grey', marker='*')
                pred_line, = ax.plot([], [], 'grey')
                pred_scat_revise = ax.scatter(pred_x_revise[0], pred_y_revise[0], s=300, c='green', marker='*')
                pred_line_revise, = ax.plot([], [], 'green')
                
            if true_gk_traces is not None:
                true_gk_team1_x = true_gk_traces[:, 0]
                true_gk_team1_y = true_gk_traces[:, 1]
                true_gk_team2_x = true_gk_traces[:, 2]
                true_gk_team2_y = true_gk_traces[:, 3]
                
                true_gk_team1_scat = ax.scatter(
                    true_gk_team1_x[0], true_gk_team1_y[0], s=team1_size, c='tab:red', alpha=0.5
                )
                true_gk_team2_scat = ax.scatter(
                    true_gk_team2_x[0], true_gk_team2_y[0], s=team2_size, c='tab:blue', alpha=0.5
                )
                
                #print(true_gk_team1_x.shape, true_gk_team1_y.shape, true_gk_team2_x.shape, true_gk_team2_y.shape)

        else:
            # if self.target_type == 'gk':
            
            if self.target_traces is not None:
                target_team1_x = self.target_traces[:, 0]
                target_team1_y = self.target_traces[:, 1]
                target_team2_x = self.target_traces[:, 2]
                target_team2_y = self.target_traces[:, 3]

                target_team1_scat = ax.scatter(target_team1_x[0], target_team1_y[0], s=team1_size, c='tab:red', alpha=0.5)
                target_team2_scat = ax.scatter(target_team2_x[0], target_team2_y[0], s=team2_size, c='tab:blue', alpha=0.5)
                target_team1_line, = ax.plot([], [], 'tab:red', alpha=0.5)
                target_team2_line, = ax.plot([], [], 'tab:blue', alpha=0.5)

            if self.pred_traces is not None:
                pred_team1_x = self.pred_traces[:, 0]
                pred_team1_y = self.pred_traces[:, 1]
                pred_team2_x = self.pred_traces[:, 2]
                pred_team2_y = self.pred_traces[:, 3]

                pred_team1_scat = ax.scatter(pred_team1_x[0], pred_team1_y[0], s=team1_size, c='tab:red')
                pred_team2_scat = ax.scatter(pred_team2_x[0], pred_team2_y[0], s=team2_size, c='tab:blue')
                pred_team1_line, = ax.plot([], [], 'tab:red')
                pred_team2_line, = ax.plot([], [], 'tab:blue')
                pred_team1_annot = ax.annotate(
                    11, xy=self.pred_traces[0, 0:2], ha='center', va='center',
                    color='w', fontsize=15, fontweight='bold', annotation_clip=False
                )
                pred_team2_annot = ax.annotate(
                    22, xy=self.pred_traces[0, 2:4], ha='center', va='center',
                    color='w', fontsize=15, fontweight='bold', annotation_clip=False
                )
        
        # annots = []
        # for i in range(len(feats)):
        #     annot = ax.annotate(feats[i][0], xy=(8, 2*(len(feats)-i)), ha='right', va='bottom', color='k', fontsize=15)
        #     annot.set_animated(True)
        #     annots.append(annot)


        def animate(t):
            
            team1_scat.set_offsets(np.dstack([team1_x[t], team1_y[t]])[0])
            team2_scat.set_offsets(np.dstack([team2_x[t], team2_y[t]])[0])
            
            if self.pred_team_poss is not None:
                team1_size = 400 + 400 * self.pred_team_poss[t, 0]
                team2_size = 400 + 400 * self.pred_team_poss[t, 1]
                team1_scat.set_sizes(np.repeat(team1_size, self.num_players // 2))
                team2_scat.set_sizes(np.repeat(team2_size, self.num_players // 2))

            t_from = max(t-19, 0)
            for i in range(self.num_players):
                player_lines[i].set_data(self.input_traces[t_from:t+1, i*4], self.input_traces[t_from:t+1, i*4+1])
                player_annots[i].set_position(self.input_traces[t, i*4:i*4+2])
                
            if self.target_type == 'ball':
                if self.target_traces is not None:
                    target_scat.set_offsets(np.array([target_x[t], target_y[t]]))
                    target_line.set_data(target_x[:t+1], target_y[:t+1])

                if self.pred_traces is not None:
                    pred_scat.set_offsets(np.array([pred_x[t], pred_y[t]]))
                    pred_line.set_data(pred_x[:t+1], pred_y[:t+1])
                    pred_scat_revise.set_offsets(np.array([pred_x_revise[t], pred_y_revise[t]]))
                    pred_line_revise.set_data(pred_x_revise[:t+1], pred_y_revise[:t+1])
                    
                # print(true_gk_team1_x[t], true_gk_team1_y[t], true_gk_team2_x[t], true_gk_team2_y[t])
                
                if true_gk_traces is not None:
                    true_gk_team1_scat.set_offsets(np.array([true_gk_team1_x[t], true_gk_team1_y[t]]))
                    true_gk_team2_scat.set_offsets(np.array([true_gk_team2_x[t], true_gk_team2_y[t]]))

            else:
                # if self.target_type == 'gk':
                
                if self.target_traces is not None:
                    target_team1_scat.set_offsets(np.array([target_team1_x[t], target_team1_y[t]]))
                    target_team2_scat.set_offsets(np.array([target_team2_x[t], target_team2_y[t]]))
                    target_team1_line.set_data(target_team1_x[:t+1], target_team1_y[:t+1])
                    target_team2_line.set_data(target_team2_x[:t+1], target_team2_y[:t+1])

                if self.pred_traces is not None:
                    pred_team1_scat.set_offsets(np.array([pred_team1_x[t], pred_team1_y[t]]))
                    pred_team2_scat.set_offsets(np.array([pred_team2_x[t], pred_team2_y[t]]))
                    pred_team1_line.set_data(pred_team1_x[:t+1], pred_team1_y[:t+1])
                    pred_team2_line.set_data(pred_team2_x[:t+1], pred_team2_y[:t+1])
                    pred_team1_annot.set_position(self.pred_traces[t, 0:2])
                    pred_team2_annot.set_position(self.pred_traces[t, 2:4])
                
            # for i, annot in enumerate(annots):
            #     annot.set_text(feats[i][t])

        frames = min(max_frames, self.input_traces.shape[0])
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200)
        plt.close(fig)
        
        return anim