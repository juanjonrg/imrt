#!/usr/bin/env python3

import os
import sys
import math
import array
import pickle
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix


class Plan:
    ''' Routines to parse and store a plan. Also calculates the dose given a fluence. '''

    name = None
    n_beams = None
    n_beamlets = None
    n_voxels = None
    dose_grid_scaling = None
    n_regions = None
    regions = None
    voxel_regions = None
    dose_matrix = None
    fluence_vector = None
    dose_vector = None
    beamlet_coordinates = None

    def __init__(self, config, rois):
        self.parse_config(config)
        self.voxel_regions = np.zeros((self.n_voxels, self.n_regions), dtype='bool')
        self.parse_voxel_rois(rois)

    def parse_config(self, config):
        self.name = config[0]
        self.n_beams = int(config[1].split()[0])
        self.n_beamlets = []
        for i in range(2, self.n_beams + 2):
            self.n_beamlets.append((int(config[i].split()[1])))
        self.n_voxels = int(config[self.n_beams + 2].split()[0])
        self.dose_grid_scaling = float(config[self.n_beams + 3].split()[0])
        self.n_regions = int(config[self.n_beams + 4].split()[0])
        self.regions = []
        for i in range(self.n_beams + 5, self.n_beams + self.n_regions + 5):
            line = config[i].split()
            self.regions.insert(int(math.log(int(line[0]), 2)), {'name': ' '.join(line[1:])})

    def __str__(self):
        lines = ['Plan summary:']
        lines.append('Name: {}'.format(self.name))
        lines.append('Number of beams: {}'.format(self.n_beams))
        lines.append('Number of beamlets:')
        for i in range(0, len(self.n_beamlets)):
            lines.append('  Beam {}: {} beamlets.'.format(i + 1, self.n_beamlets[i]))
        lines.append('Number of voxels: {}'.format(self.n_voxels))
        lines.append('Dose Grid Scaling: {}'.format(self.dose_grid_scaling))
        lines.append('Number of regions: {}'.format(self.n_regions))
        for i in range(len(self.regions)):
            lines.append('  Region {:2} ({:4}): {:15} {:8} voxels.'.format(
                i, int(math.pow(2, i)), self.regions[i]['name'], self.regions[i]['n_voxels']))
            if self.dose_matrix is None:
                lines.append('No dose deposition matrix')
        else:
            lines.append('Dose deposition matrix: {} x {} with {} nonzeroes.'.format(
                self.dose_matrix.shape[0], self.dose_matrix.shape[1], self.dose_matrix.nnz))
        if self.fluence_vector is None:
                lines.append('No fluence vector')
        else:
            lines.append('Fluence vector: {} beamlets.'.format(len(self.fluence_vector)))
        if self.dose_vector is None:
            lines.append('No dose vector')
        else:
            lines.append('Dose vector: {} voxels.'.format(len(self.dose_vector)))
        return '\n'.join(lines)

    def int_2_bool_list(self, n):
        return [x == '1' for x in format(n, '0' + str(self.n_regions) + 'b')[::-1]]

    def parse_voxel_rois(self, rois):
        i = 0        
        for v in rois:
            self.voxel_regions[i, :] = self.int_2_bool_list(int(v.split()[0]))
            i += 1

        for i in range(len(self.regions)):
            self.regions[i]['n_voxels'] = np.sum(self.voxel_regions[:, i])

    def compute_dose(self, fluence=None):
        if fluence is None:
            fluence = self.fluence_vector
        self.dose_vector = self.dose_matrix.dot(fluence) * self.dose_grid_scaling

        for i in range(len(self.regions)):
            self.regions[i]['doses'] = []

        for i in range(len(self.dose_vector)):
            dose = float(self.dose_vector[i])
            for r in range(len(self.regions)):
                if self.voxel_regions[i, r]:
                    self.regions[r]['doses'].append(dose)


def stats(rids=None):
    ''' Calculates max, min, avg dose for the given region ids (or all if None) '''
    lines = [];
    header = "{:^16} {:^11.6} {:^11.6} {:^11.6}".format('Region', 'Min', 'Avg', 'Max')
    lines.append('-'*len(header))
    lines.append(header)
    lines.append('-'*len(header))
    if rids is None:
        rids = list(range(len(plan.regions)))
    for r in rids:
        region = plan.regions[r]
        doses = region['doses']
        lines.append("{:<16} {:>11.6f} {:>11.6f} {:>11.6f}".format(region['name'], np.min(doses), np.average(doses), np.max(doses)))
    return lines

def plot(filename=None, rids=None):
    ''' Plots the Dose Volume Histogram '''
    for line in stats(rids=rids):
        print(line)
    #plt.rcParams['figure.figsize'] = [18, 10]
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 18
    bins = 1000
    max_gy = 100
    interval = max_gy/bins
    if rids is None:
        rids = list(range(len(plan.regions)))
    for r, color in rids.items():
        region = plan.regions[r]
        hist = np.histogram(region['doses'], bins=bins, range=(0, max_gy))
        cumsum = (len(region['doses']) - np.insert(np.cumsum(hist[0]), 0, 0))
        cumsum = cumsum*100/len(region['doses'])
        plt.plot(hist[1], cumsum, linewidth=3, label=region['name'], color=color)

    plt.legend()
    plt.grid()
    #plt.title(plan.name)
    plt.ylabel("Volume (%)")
    plt.xlabel("Dose (Grays)")
    plt.xticks(np.arange(0,105,5))
    plt.yticks(np.arange(0,105,10))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def load_plan(folder):
    ''' Opens and loads the plan files. Only tested on Linux! ''' 
    files = os.listdir(folder)

    matching = fnmatch.filter(files, 'm_PARETO*')
    print("Loading", matching[0], '...')
    with open(os.path.join(folder, matching[0])) as f: 
        config = f.read().splitlines()

    matching = fnmatch.filter(files, 'v_PARETO*')
    print("Loading", matching[0], '...')
    with open(os.path.join(folder, matching[0])) as f: 
        voxel_rois = f.read().splitlines()

    plan = Plan(config, voxel_rois)

    rows = array.array('i')
    cols = array.array('i')
    vals = array.array('i')
    col_offset = 0
    beam_index = 0
    matching = fnmatch.filter(files, 'd_PARETO*')
    for filename in sorted(matching):
        print("Loading", filename, '...')
        with open(os.path.join(folder, filename)) as f: 
            next(f)
            for line in f:
                line = [int(x) for x in line.split()]
                rows.append(line[0])
                cols.append(line[1] + col_offset)
                vals.append(line[2])
            col_offset += plan.n_beamlets[beam_index]
            beam_index += 1
    plan.dose_matrix = coo_matrix((vals, (rows, cols)), shape=(plan.n_voxels, sum(plan.n_beamlets)), dtype=np.int32).tocsr()

    matching = fnmatch.filter(files, 'x_PARETO*')
    fluence = []
    for filename in sorted(matching):
        print("Loading", filename, '...')
        with open(os.path.join(folder, filename)) as f: 
            fluence += [float(x.split()[-1]) for x in f.readlines()]
    plan.fluence_vector = fluence

    matching = fnmatch.filter(files, 'xcoords_PARETO*')
    plan.beamlet_coordinates = []
    for filename in sorted(matching):
        print("Loading", filename, '... ', end='')
        with open(os.path.join(folder, filename)) as f:
            coords = []
            for line in f.readlines()[12:]:
                line = line.split()
                coords.append((int(line[0]), int(line[1])))
            plan.beamlet_coordinates.append(coords)
            print(len(coords), 'coordinates.')

    plan.compute_dose()
    print()
    print(plan)
    return plan

def plot_beam(index, fluence, title, filename=None):
    ''' Plots the BEV of a given beam by index '''
    start = sum(plan.n_beamlets[:index])

    beam = np.zeros((120, 120))
    for i in range(plan.n_beamlets[index]):
        x = plan.beamlet_coordinates[index][i][0]
        y = plan.beamlet_coordinates[index][i][1]
        beam[x][y] = fluence[start + i]

    plt.style.use("default")
    plt.rcParams['figure.figsize'] = [18, 10]
    plt.rcParams['font.size'] = 18
    plt.matshow(beam, vmin=0, vmax=0.2, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title(title, y=1.2)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def translate_names(plan):
    ''' Translates the names of the regions (only for the papers, polski is cool) '''

    translations = {
            'BODY': 'Body',
            'pien mozgu': 'Brainstem',
            'pien mozgu+3mm': 'Brainstem +3mm',
            'rdzen kreg.': 'Spinal cord',
            'rdzen kreg.+3mm': 'Spinal cord +3mm',
            'slinianka L': 'Salivary gland L',
            'slinianka P': 'Salivary gland R',
            'zuchwa': 'Jaw',
            'NT': 'Normal tissue'
            }
    for region in plan.regions:
        if region['name'] in translations:
            region['name'] = translations[region['name']]

def regions_to_plot(plan):
    ''' Fixes the colors of each region, so different patients use the same colors.
    This function is awful and uses the english labels to identify regions. We should use an id!'''

    regions = {
            'Salivary gland L': '#37a134', 
            'Salivary gland R': '#3374b2', 
            'Brainstem +3mm': '#fa7e21', 
            'Spinal cord +3mm': '#38bdcd',
            'Jaw': '#df72c0', 
            'Normal tissue': '#7e7e7e' 
            }
    ptv_colors = ['#d0272f', '#89564d', '#9363bb']

    rids = {}
    for r, color in regions.items():
        for i in range(len(plan.regions)):
            if plan.regions[i]['name'] == r:
                rids[i] = color 
    last_ptv_color = 0
    for r in range(len(plan.regions)):
        name = plan.regions[r]['name']
        #if name in regions:
        #    rids[r] = regions[name]
        if 'PTV' in name:
            rids[r] = ptv_colors[last_ptv_color]
            last_ptv_color += 1
    return rids

if __name__ == '__main__':

    plan = load_plan('/home/juanjo/repos/Radiotherapy/plans/5') # Folder with all the plan files

    # Folder with the plan results you want to plot. 
    # It looks for a pickle file with the fluence, like the ones I uploaded to the _extra zips in RaaS.
    result_folder = '/home/juanjo/repos/Radiotherapy/results/gurobi/x_5_LPS122_20200928'
    files = os.listdir(result_folder)
    matching = fnmatch.filter(files, '*.pkl')
    with open(os.path.join(result_folder, matching[0]), 'rb') as f:
        fluence = pickle.load(f)

    plan.compute_dose(fluence)
    translate_names(plan)
    #plot(filename=os.path.join(result_folder, 'DVH.pdf'), rids=regions_to_plot(plan))
    plot(filename=os.path.join(result_folder, 'DVH.png'), rids=regions_to_plot(plan))

    with open(os.path.join(result_folder, 'stats.txt'), 'w') as f:
        for line in stats(rids=regions_to_plot(plan)):
            f.write(line + '\n')

    for index in range(plan.n_beams):
        plot_beam(index, fluence, 'Beam {}'.format(index + 1), '{}/Beam_{}.png'.format(result_folder, index + 1))
