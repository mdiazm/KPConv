""" Class to load LiDAR point clouds (in ply format) and use them as input to this KPConv Framework """

# Basic libs
import json
import os
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree

# PLY files reader
from utils.ply import read_ply, write_ply

# OS functins
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

# Dataset parent class
from datasets.common import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling as cpp_subsampling

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)

# Class definition

class LiDARDataset(Dataset):
    """
    Class to handle LiDAR point clouds for segmentation task.
    """

    def __init__(self, input_threads=8):
        Dataset.__init(self, 'LiDAR')

        self.label_to_names = {
            0: 'unlabeled',
            1: 'man-made terrain',
            2: 'natural terrain',
            3: 'high vegetation',
            4: 'low vegetation',
            5: 'buildings',
            6: 'hard scape',
            7: 'scanning artefacts',
            8: 'cars'
        }

        # Initialize variables concerning class labels
        self.init_labels()

        # List of classes ignored during training
        self.ignored_labels = np.sort([0]) ## TODO when this dataset is used for training, change this parameter

        """
            Parameters of the files
        """
        # Path of the folder containing ply files
        self.path = 'Data/LiDAR'

        # Original data path - only for .ply training files (without being subsampled!)
        self.original_folder = 'original_data'

        # Path of the files
        self.train_path = join(self.path, 'train')
        self.test_path = join(self.path, 'test')

        # Proportion of validation scenes
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.validation_split = 5

        # LiDAR files
        self.lidar_files = {

        }

        """
        Prepare PLY files
        """
        self.prepare_data()

        # List of training and test files
        self.train_files = np.sort([join(self.train_path), f] for f in listdir(self.train_path) if f[-4:] == '.ply') # f[-4:] returns last 4 characters of the file name (extension)
        self.test_files = np.sort([join(self.test_path), f] for f in listdir(self.test_path) if f[-4] == '.ply')

    def prepare_data(self):
        """
        Prepare LiDAR (.ply) files
        """

        if not exists(self.train_path):
            makedirs(self.train_path)
        if not exists(self.test_path):
            makedirs(self.test_path)

        # Folder names
        old_folder = join(self.path, self.original_folder)
        train_folder = join(old_folder, 'train')
        test_folder = join(old_folder, 'test')

        # Generate train files (subsample clouds)

        # PLY files containing points
        cloud_names = [file_name[:-4] for file_name in listdir(train_folder)]
        for cloud_name in cloud_names:

            # Name of the input file
            input_file = join(old_folder, cloud_name + '.ply')

            # Name of the output file
            ply_file_full = join(self.train_path, cloud_name + '.ply')

            # Pass if this cloud was processed yet
            if exists(ply_file_full):
                print("{:s} already done\n".format(cloud_name))
                continue

            print("Preparing {:s}".format(cloud_name))

            data = read_ply(input_file)

            # Points
            x = np.asarray(data['x'], dtype=np.float)
            y = np.asarray(data['y'], dtype=np.float)
            z = np.asarray(data['z'], dtype=np.float)

            points = np.column_stack((x, y, z))

            # Colors
            r = np.asarray(data['red'], dtype=np.float)
            g = np.asarray(data['green'], dtype=np.float)
            b = np.asarray(data['blue'], dtype=np.float)

            colors = np.column_stack((r, g, b))

            # Labels
            labels = np.asarray(data['scalar_Classification'], dtype=np.int32)

            # Subsample file
            sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                  features=colors,
                                                                  labels=labels,
                                                                  sampleDl=0.01)

            # Write subsampled file
            write_ply(
                ply_file_full, # File where data is going to be written
                (sub_points, sub_colors, sub_labels), # Data to be written
                ['x', 'y', 'z', 'rec', 'green', 'blue', 'class'] # Headers of the data to be written
            )

        # Generate test files (change ply format, deleting unnecesary fields)
        cloud_names = [file_name[:-4] for file_name in listdir(test_folder)]
        for cloud_name in cloud_names:

            # Name of the input file
            input_file = join(old_folder, cloud_name + '.ply')

            # Name of the output file
            ply_file_full = join(self.train_path, cloud_name + '.ply')

            # Pass if this cloud was processed yet
            if exists(ply_file_full):
                print("{:s} already done\n".format(cloud_name))
                continue

            print("Preparing {:s}".format(cloud_name))

            data = read_ply(input_file)

            # Points
            x = np.asarray(data['x'], dtype=np.float)
            y = np.asarray(data['y'], dtype=np.float)
            z = np.asarray(data['z'], dtype=np.float)

            points = np.column_stack((x, y, z))

            # Colors
            r = np.asarray(data['red'], dtype=np.float)
            g = np.asarray(data['green'], dtype=np.float)
            b = np.asarray(data['blue'], dtype=np.float)

            colors = np.column_stack((r, g, b))

            # Write subsampled file
            write_ply(
                ply_file_full,  # File where data is going to be written
                (points, colors),  # Data to be written
                ['x', 'y', 'z', 'rec', 'green', 'blue']  # Headers of the data to be written
            )

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches)
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm)')

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        if not exists(tree_path):
            makedirs(tree_path)

        # All training and test files
        files = np.hstack((self.train_files, self.test_files))

        # Initialize containers
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Advanced display
        N = len(files)
        progress_n = 30 # Total number of files
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nPreparing KDTree for all scenes, subsampled at: {:.3f}'.format(subsampling_parameter))

        for i, file_path in enumerate(files):

            # Restart timer
            t0 = time.time()

            # Get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            if 'train' in cloud_folder:
                if self.all_splits[i] == self.validation_split:
                    cloud_split = 'validation'
                else:
                    cloud_split = 'training'
            else:
                cloud_split = 'test'

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if isfile(KDTree_file):

                # Read PLY with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']

                # Read PKL with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                # Calculate KDTree from subsampled point clouds again

                # Read PLY file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    int_features = None
                else:
                    int_features = data['class']

                # Subsample cloud
                sub_data = grid_subsampling(points,
                                            features=colors,
                                            labels=int_features,
                                            sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_data[1] / 255

                # Get chosen neighborhoods
                search_tree = KDTree(sub_data[0], leaf_size=50)

                # Save computed KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save PLY
                if cloud_split == 'test':
                    sub_labels = None
                    write_ply(sub_ply_file,
                              [sub_data[0], sub_colors],
                              ['x', 'y', 'z', 'red', 'green', 'blue'])
                else:
                    sub_labels = np.squeeze(sub_data[2])
                    write_ply(sub_ply_file,
                              [sub_data[0], sub_colors, sub_labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

            print('', end='\r')
            print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])
        self.num_test = len(self.input_trees['test'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        self.test_proj = []
        self.test_labels = []
        i_val = 0
        i_test = 0

        # Advanced display
        N = self.num_validation + self.num_test
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # Get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            # Validation projection and labels
            if 'train' in cloud_folder and self.all_splits[i] == self.validation_split:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:

                    # Get the original points
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']

                    # Compute projection inds. Get indices of all the points. For each training set.
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.validation_proj += [proj_inds]
                self.validation_labels += [labels]
                i_val += 1

            # Test projection
            if 'test' in cloud_folder:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:

                    # Get original points
                    full_ply_path = file_path.split('/')
                    full_ply_path = join(self.test_path, full_ply_path[-1])
                    # full_ply_path = '/'.join(full_ply_path)
                    data = read_ply(full_ply_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = np.zeros(points.shape[0], dtype=np.int32)

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(points), return_distance=False)
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.test_labels += [labels]
                i_test += 1

            print('', end='\r')
            print(fmt_str.format('#' * (((i_val + i_test) * progress_n) // N), 100 * (i_val + i_test) / N),
                  end='',
                  flush=True)

        print('\n')

        return

    def get_batch_gen(self, split, config):
        """
        Function defining the batch generator for each split. Should return the generator, the generated types and the
        generated shapes.
        :param split: string in training, validation or test.
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        # Parameters #
        if split == 'training':

            # First compute the number of points we want to pick in each cloud and for each class
            epoch_n = config.epoch_steps * config.batch_num
            random_pick_n = int(np.ceil(epoch_n / (self.num_training * config.num_classes)))

        elif split == 'validation':

            # First compute the number of points we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        elif split == 'test':

            # First compute the number of points we want to pick for each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        elif split == 'ERF':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = 1000000
            self.batch_limit = 1
            np.random.seed(42)
        else:
            raise ValueError('Split argument in data generator should be training, validation or test')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}

        # Reset potentials
        self.potentials[split] = []
        self.min_potentials[split] = []
        data_split = split

        if split == 'ERF':
            data_split = 'test'

        for i, tree in enumerate(self.input_trees[data_split]):
            self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]

        # Define generators functions
        def gen_random_epoch_inds():

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int32)

            # Choose random points of each class for each cloud
            for cloud_ind, cloud_labels in enumerate(self.input_labels[split]):
                epoch_indices = np.empty((0,), dtype=np.int32)
                for label_ind, label in enumerate(self.label_values):
                    if label not in self.ignored_labels:

                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        if len(label_indices) <= random_pick_n:
                            epoch_indices = np.hstack((epoch_indices, label_indices))
                        elif len(label_indices) < 50 * random_pick_n:
                            new_randoms = np.random.choice(label_indices, size=random_pick_n, replace=False)
                            epoch_indices = np.hstack((epoch_indices, new_randoms.astype(np.int32)))
                        else:
                            rand_inds = []
                            while len(rand_inds) < random_pick_n:
                                rand_inds = np.unique(np.random.choice(label_indices, size=5*random_pick_n, replace=True))
                            epoch_indices = np.hstack((epoch_indices, rand_inds[:random_pick_n].astype(np.int32)))

                # Stack those indices with the cloud index
                epoch_indices = np.vstack((np.full(epoch_indices.shape, cloud_ind, dtype=np.int32), epoch_indices))

                # Update the global indice container
                all_epoch_inds = np.hstack((all_epoch_inds, epoch_indices))

            return all_epoch_inds