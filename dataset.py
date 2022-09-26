import os
import h5py
import numpy as np
import pandas as pd
import random
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from target_poss_encoder import TargetPossEncoder


class SoccerDataset(Dataset):
    def __init__(self, trace_files, mode='true_gk',
                 trace_dir="./", heatmap_dir="./data/fm_heatmaps",
                 interpolate=False, heatmap_compute=False, team_separate=True, poss_encode=False, augment=True, masking=False,
                 window_size=50, pitch_size=(108, 72), sigma=2):

        self.mode = mode
        self.masking = masking
        self.interpolate = interpolate
        self.heatmap_compute = heatmap_compute
        self.team_separate = team_separate
        self.poss_encode = poss_encode
        self.augment = augment
        self.random_permute = True

        
        self.window_size = window_size
        self.pitch_size = pitch_size

        if self.mode == 'no_gk':
            num_players = 10
            target_cols = ['A11_x', 'A11_y', 'B11_x', 'B11_y']
        else:
            # i.e. self.mode == 'true_gk' or self.mode == 'pred_gk':
            num_players = 11
            target_cols = ['ball_x', 'ball_y']
            
        self.num_players = num_players

        input_cols_x = [f'{team}{i:02d}_x' for team in ['A', 'B'] for i in np.arange(num_players) + 1]
        input_cols_y = [f'{team}{i:02d}_y' for team in ['A', 'B'] for i in np.arange(num_players) + 1]
        input_cols = np.vstack((input_cols_x, input_cols_y)).ravel('F').tolist()
        
        trace_list = []
        heatmap_list = []
        
        for trace_file in trace_files:
            session_name = trace_file.split('.')[0]
            session_trace_df = pd.read_csv(trace_dir + trace_file, header=0, index_col=0, dtype=np.float32)
            session_trace_df['session'] = session_name

            trace_list.append(session_trace_df[['session'] + session_trace_df.columns[:-1].tolist()])

            if self.heatmap_compute:
                heatmap_file = f'{session_name}.npy'
                heatmap_path = f'{heatmap_dir}_{mode}/{heatmap_file}'

                if not os.path.exists(heatmap_path):
                    input_traces = session_trace_df[input_cols].values
                    instant_heatmap_list = []
                    tqdm_iter = tqdm(range(session_trace_df.shape[0]), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
                    for i in tqdm_iter:
                        instant_heatmaps = self.compute_heatmaps(input_traces[i], sigma)
                        instant_heatmap_list.append(instant_heatmaps)
                    heatmaps = np.array(instant_heatmap_list)
                    np.save(heatmap_path, heatmaps)
                    print(f"Heatmaps saved in '{heatmap_path}'.")
                else:
                    heatmaps = np.load(heatmap_path)
                    print(f"Heatmaps loaded from '{heatmap_path}'")

                heatmap_list.append(heatmaps)

        trace_df = pd.concat(trace_list, ignore_index=True)
        
        episode_sizes = trace_df['episode'].value_counts()
        trace_df = trace_df[trace_df['episode'].isin(episode_sizes[episode_sizes > window_size].index)]
        trace_df['episode'] = trace_df['episode'].diff().fillna(0).clip(upper=1).cumsum().astype(int).values
        self.bins = [0] + (trace_df['episode'].value_counts(sort=False) - window_size + 1).cumsum().tolist()
                
        input_cols_speed = [f'{team}{i:02d}_speed' for team in ['A', 'B'] for i in np.arange(num_players) + 1]
        input_cols_accel = [f'{team}{i:02d}_accel' for team in ['A', 'B'] for i in np.arange(num_players) + 1]
        new_input_cols = np.vstack((input_cols_x, input_cols_y, input_cols_speed, input_cols_accel)).ravel('F').tolist()
        
        self.input_traces = trace_df[new_input_cols].values.astype('float32')
        self.target_traces = trace_df[target_cols].values.astype('float32')
        
        if self.heatmap_compute:
            self.input_heatmaps = np.concatenate(heatmap_list, axis=0).astype('float32')[trace_df.index]
        else:
            self.input_heatmaps = None

        # with h5py.File(h5_path, 'w') as f:
        #     f.create_dataset('input_trace', data=input_trace)
        #     f.create_dataset('target_trace', data=target_trace)
        #     f.create_dataset('input_heatmaps', data=input_heatmaps)
        

        if self.poss_encode:
            self.tpe = TargetPossEncoder(trace_df[['session', 'episode'] + new_input_cols + ['ball_x', 'ball_y']].values, self.pitch_size)
            self.tpe.run()
            self.target_player_poss = self.tpe.player_poss
            self.target_team_poss = self.tpe.team_poss

    def __len__(self):
        self.input_traces.shape[0] - self.window_size + 1
        return self.bins[-1]

    def __getitem__(self, i):
        episode = pd.cut([i], self.bins, right=False, labels=np.arange(len(self.bins) - 1))[0]
        idx_from = i + episode * (self.window_size - 1)
        idx_to = idx_from + self.window_size


        input_traces = self.input_traces[idx_from:idx_to]
        input_heatmaps = None

        target_traces = self.target_traces[idx_from:idx_to]
        target_player_poss = None
        target_team_poss = None
        # target_trace_decoded = None
        # pitch_mask = None
        items = [input_traces, target_traces]


        if self.masking:
            random_mask = np.random.randint(0, [100, 22], (110,2))
            input_traces[random_mask[:,0], random_mask[:,1]*2] = 0
            input_traces[random_mask[:,0], random_mask[:,1]*2+1] = 0
            
            
        # first_is_a = input_traces.reshape(self.window_size, -1, 4)[:,:self.num_players,0].mean() < input_traces.reshape(self.window_size, -1, 4)[:,self.num_players:,0].mean()
            
        if self.random_permute:
            if random.random():
                permutation = np.concatenate([np.random.permutation(self.num_players), np.random.permutation(self.num_players) + self.num_players], -1)
            else:
                target_traces = target_traces[:,np.array([2,3,0,1])]
                permutation = np.concatenate([np.random.permutation(self.num_players) + self.num_players, np.random.permutation(self.num_players)], -1)

            input_traces = input_traces.reshape(self.window_size, -1, 4)[:,permutation].reshape(self.window_size, -1)

        items = [input_traces, target_traces]
        if self.heatmap_compute:
            input_heatmaps = self.input_heatmaps[idx_from:idx_to]
            items.append(input_heatmaps)
        if self.poss_encode:
            target_player_poss = self.target_player_poss[idx_from:idx_to]
            target_team_poss = self.target_team_poss[idx_from:idx_to]
            items.extend([target_player_poss, target_team_poss])
            
        if self.augment:
            return self.xy_flip_augment(
                input_traces, target_traces, input_heatmaps, target_player_poss, target_team_poss
            )
        else:
            return items

    def compute_heatmaps(self, traces, sigma):
        num_players = traces.shape[0] // 2
        pitch_range = [[0, self.pitch_size[0]], [0, self.pitch_size[1]]]
        
        if not self.team_separate:
            x = traces[0::2]
            y = traces[1::2]
            heatmap = np.histogram2d(x, y, bins=self.pitch_size, range=pitch_range)[0]
            heatmap = gaussian_filter(heatmap, sigma=sigma)
            return heatmap.unsqueeze(0)

        else:
            team1_x = traces[0:num_players:2]
            team1_y = traces[1:num_players:2]
            team2_x = traces[num_players::2]
            team2_y = traces[num_players+1::2]
            heatmap1 = np.histogram2d(team1_x, team1_y, bins=self.pitch_size, range=pitch_range)[0]
            heatmap1 = gaussian_filter(heatmap1, sigma=sigma)
            heatmap2 = np.histogram2d(team2_x, team2_y, bins=self.pitch_size, range=pitch_range)[0]
            heatmap2 = gaussian_filter(heatmap2, sigma=sigma)
            return np.stack([heatmap1, heatmap2])

    def xy_flip_augment(self, input_traces, target_traces,
                        input_heatmaps=None, target_player_poss=None, target_team_poss=None):
        flip_x = random.randint(0, 1)
        flip_y = random.randint(0, 1)
        
        target_interval = 2 if self.mode == 'no_gk' else 4

        if flip_x or flip_y:
            # Random flip by returning (reference - coordinate) * multiplier
            # For flip = 0, set ref = 0 and mul = -1. i.e., return (0 - coord) * (-1) = coord
            # For flip = 1, set ref = pitch_lim and mul = 1. i.e., return (pitch_lim - coord) * 1

            input_ref = np.zeros(input_traces.shape).astype('float32')          # ref for input_trace with flip = 0
            input_mul = np.full(input_traces.shape, -1).astype('float32')       # mul for input_trace with flip = 0
            target_ref = np.zeros(target_traces.shape).astype('float32')        # ref for target_trace with flip = 0
            target_mul = np.full(target_traces.shape, -1).astype('float32')     # mul for target_trace with flip = 0

            if flip_x:
                input_ref[:, 0::4] = self.pitch_size[0]
                input_mul[:, 0::4] = 1
                target_ref[:, 0::target_interval] = self.pitch_size[0]
                target_mul[:, 0::target_interval] = 1
            if flip_y:
                input_ref[:, 1::4] = self.pitch_size[1]
                input_mul[:, 1::4] = 1
                target_ref[:, 1::target_interval] = self.pitch_size[1]
                target_mul[:, 1::target_interval] = 1

            input_traces = (input_ref - input_traces) * input_mul
            target_traces = (target_ref - target_traces) * target_mul

            if self.heatmap_compute:
                flip_axis = []
                if flip_x:  # Flip the 2nd axis of input_heatmaps
                    flip_axis.append(2)
                if flip_y:  # Flip the 3rd axis of input_heatmaps
                    flip_axis.append(3)
                input_heatmaps = np.flip(input_heatmaps, tuple(flip_axis)).copy()

#             if self.encode:
#                 target_encoded = target_encoded.copy()

#                 if self.encode_mode == 'poss':
#                     cor_ll = self.num_players + 1
#                     cor_lr = self.num_players + 2
#                     cor_ul = self.num_players + 3
#                     cor_ur = self.num_players + 4

#                     if flip_x:  # Swap the columns of the left and right corners
#                         player_poss[:, [cor_ll, cor_lr]] = player_poss[:, [cor_lr, cor_ll]]
#                         player_poss[:, [cor_ul, cor_ur]] = player_poss[:, [cor_ur, cor_ul]]
#                     if flip_y:  # Swap the columns of the upper and lower corners
#                         player_poss[:, [cor_ll, cor_ul]] = player_poss[:, [cor_ul, cor_ll]]
#                         player_poss[:, [cor_lr, cor_ur]] = player_poss[:, [cor_ur, cor_lr]]

#                     target_trace_decoded = (target_ref - target_trace_decoded) * target_mul
                
#                 elif self.encode_mode == 'state':
#                     if flip_x:
#                         player_poss[:, [3, 4]] = player_poss[:, [4, 3]]
#                     if flip_y:
#                         player_poss[:, [5, 6]] = player_poss[:, [6, 5]]

        items = [input_traces, target_traces]
        for item in [input_heatmaps, target_player_poss, target_team_poss]:
            if item is not None:
                items.append(item)
        return items