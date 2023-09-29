# coding=utf-8

import argparse
import os

import numpy as np

from ravens import Dataset
from ravens import Environment
from ravens import tasks
import pdb
import pybullet as p

from v11_heuristic_packing import *
import trimesh

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--disp', default=False, type=str2bool)
    parser.add_argument('--task', default='simulate-irregular')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--stable', default=False, type=str2bool)
    parser.add_argument('--method', default='sdf', choices=['dblf','hm','mta','sdf','random', 'first','genetic'])
    parser.add_argument('--n', default=10, type=int)
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--case_cnt', default=1000, type=int)
    parser.add_argument('--contr_seq', default=True, type=str2bool)
    parser.add_argument('--vol_dec', default=False, type=str2bool)
    parser.add_argument('--cuda', default=False, type=str2bool)
    parser.add_argument('--record', default=False, type=str2bool)
    parser.add_argument('--sdf_remain_terms', default='1234')
    args = parser.parse_args()

    # Initialize environment and task.
    env = Environment(args.disp, hz=480)
    task = tasks.names[args.task]()
    task.mode = args.mode
    kernal = compute_sdf_kernal(args.cuda)

    # Initialize scripted oracle agent and dataset.
    dataset = Dataset(os.path.join('data', f'{args.task}-{task.mode}'))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
      seed = -1 if (task.mode == 'test') else -2
    
  # Collect training data from oracle demonstrations.
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{args.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    data_dir = os.path.join('.', 'dataset', 'our_oriented_dataset')
    occ_dir = os.path.join('.', 'dataset', 'our_oriented_occs')
    avg_success_cnt = 0.0
    avg_success_rate = 0.0
    avg_compactness = 0.0
    avg_pyramidality = 0.0
    avg_time = 0.0
    avg_volume = 0.0
    import csv
    if args.contr_seq and not args.vol_dec:
      if args.method == 'sdf' and args.sdf_remain_terms != '1234':
        csvname = 'results_'+get_method_name(args.method)+'_sdf_rem_'+args.sdf_remain_terms+'_reorder_80.csv'
      else:
        csvname = 'results_'+get_method_name(args.method)+'_reorder_80.csv'
    elif args.contr_seq and args.vol_dec:
      if args.method == 'sdf' and args.sdf_remain_terms != '1234':
        csvname = 'results_'+get_method_name(args.method)+'_sdf_rem_'+args.sdf_remain_terms+'_voldec_80.csv'
      else:
        csvname = 'results_'+get_method_name(args.method)+'_voldec_80.csv'
    elif not args.contr_seq:
      if args.method == 'sdf' and args.sdf_remain_terms != '1234':
        csvname = 'results_'+get_method_name(args.method)+'_sdf_rem_'+args.sdf_remain_terms+'_fixed_80.csv'
      else:
        csvname = 'results_'+get_method_name(args.method)+'_fixed_80.csv'
    if args.record:
        if not os.path.isfile(csvname):
        	with open(csvname, 'w', newline='') as f:
        		fieldnames = ['packing_case_id', 'success_cnt',  'success_rate', 'compactness', 'pyramidality', 'time_per_obj', 'volume']
        		writer = csv.DictWriter(f, fieldnames=fieldnames)
        		writer.writeheader()
        		for c_ in range(1000):
        			writer.writerow({'packing_case_id': c_, 'success_rate':None, 'compactness':None, 'pyramidality':None, 'success_cnt':None, 'time_per_obj':None, 'volume':None})
    for shape_code_idx in range(args.split*args.case_cnt, (args.split+1)*args.case_cnt):
        start_ = time.time()
        print ()
        print ('Testing Packing Case', shape_code_idx)
        model = GeoBin(x_max=32, y_max=32, z_max=30, thickness=2)
        obs, reward, _, info = env.reset_packing(task)

        # Loading objects
        tot_obj_num = 80
        shape_codes = np.load('1000_packing_sequences_of_80_objects.npy')[shape_code_idx][0:tot_obj_num]
        max_num_vertices = 0
        max_num_faces = 0
        dict_obj_occ = {}
        for i in range(len(shape_codes)):
            obj_type = shape_codes[i] # 69
            print ('item', i, 'loading type', obj_type)
            # Load the Mesh
            obj_filename = os.path.join(data_dir, np.sort(os.listdir(data_dir))[obj_type])
            print (obj_filename)
            if not os.path.isfile(obj_filename):
            	print ('Cannot find', obj_filename)
            obj = trimesh.load_mesh(obj_filename, file_type='ply', process=False)
            # Build the compose_occ for single objects
            for r in range(4):
                dict_obj_occ[str(i) + '_r' + str(r)] = np.load(os.path.join(occ_dir, os.path.basename(obj_filename)[:-4]+'_objocc.npy'), allow_pickle=True).item()['r' + str(r)]
        
        p.setGravity(0,0,-9.80665)
        
        list_placed = []
        _,_,_,tot_time, packed_volumes, scene_top_down = sdf_placement_couple_reorder_concave_with_simulation(model, kernal, shape_codes, env, dict_obj_occ, require_stability=args.stable, choose_objective=args.method, use_cuda=args.cuda, sdf_remain_terms=args.sdf_remain_terms, vol_dec=args.vol_dec)
        
        success_cnt = 0
        packed_volume = 0.0
        i = -1
        for k in env.obj_ids['rigid']:
            i += 1
            if p.getBasePositionAndOrientation(k)[0][0] < 0.5-0.33/2 or p.getBasePositionAndOrientation(k)[0][0] > 0.5+0.33/2 or \
                p.getBasePositionAndOrientation(k)[0][1] < -0.33/2 or p.getBasePositionAndOrientation(k)[0][1] > 0.33/2 or \
                p.getBasePositionAndOrientation(k)[0][2] < 0 or p.getBasePositionAndOrientation(k)[0][2] > 0.30:
                success_cnt += 0
            else:
                success_cnt += 1
                packed_volume += packed_volumes[i]
        print ('Success Cnt', success_cnt)
        print ('Time Spent:', time.time() - start_)
        avg_success_cnt += success_cnt/args.case_cnt
        avg_success_rate += success_cnt/tot_obj_num/args.case_cnt
        avg_compactness += packed_volume/scene_top_down.max()/32/32*5/args.case_cnt
        avg_pyramidality += packed_volume/scene_top_down.sum()*5/args.case_cnt
        avg_time += tot_time/success_cnt/args.case_cnt
        avg_volume += packed_volume/args.case_cnt
        print ('avg_success_rate', avg_success_rate/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('avg_compactness', avg_compactness/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('avg_pyramidality',avg_pyramidality/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('average_success_cnt', avg_success_cnt/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('average_time_per_obj', avg_time/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('average_volume', avg_volume/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)

        if args.record:
            with open(csvname, newline='') as f:
               data = [row for row in csv.DictReader(f)]
               data[shape_code_idx]['success_cnt'] = success_cnt
               data[shape_code_idx]['success_rate'] = success_cnt/tot_obj_num
               data[shape_code_idx]['compactness'] = packed_volume/(scene_top_down.max()*32*32/5)
               data[shape_code_idx]['pyramidality'] = packed_volume/(scene_top_down.sum()/5)
               data[shape_code_idx]['time_per_obj'] = tot_time/success_cnt
               data[shape_code_idx]['volume'] = packed_volume
               header = data[0].keys()
            with open(csvname, 'w', newline='') as f:
               writer = csv.DictWriter(f, fieldnames=header)
               writer.writeheader()
               writer.writerows(data)

if __name__ == '__main__':
  main()
