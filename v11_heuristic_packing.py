import torch
import trimesh
from trimesh.transformations import translation_matrix, concatenate_matrices, rotation_matrix, scale_matrix, decompose_matrix
from PIL import Image
import numpy as np
from torch.autograd import Variable
import datetime
import os
import shutil
import pdb
import time
import glob
import pdb
import copy
import sys
import cv2

from geo_model_v10 import *
import torch
import torch.nn.functional as F
import pybullet as p

def compute_sdf_kernal(use_cuda=False):
	x = torch.arange(10+1)
	y = torch.arange(10+1)
	z = torch.arange(10*5+1)
	z_scaling = torch.tensor([1.0,1.0,0.2])
	center = torch.tensor([[5.0,5.0,25.0]])
	grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
	grid_pts = torch.cat((grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)), dim=-1).unsqueeze(1).unsqueeze(0)
	kernal = torch.max(torch.abs((grid_pts - center))*z_scaling.unsqueeze(0).unsqueeze(1).unsqueeze(2), dim=-1)[0].reshape(11,11,10*5+1)
	kernal = torch.ceil(kernal)
	if use_cuda:
		print ("Using CUDA")
		kernal = kernal.cuda()
	return kernal

def packing_scene_initialize(model):
	scene_top_down = np.zeros((
		int(model.x_max / model.voxel_resolution/5),
		int(model.y_max / model.voxel_resolution/5))).astype('int')
	scene_occ = np.zeros((
		int(model.x_max / model.voxel_resolution/5),
		int(model.y_max / model.voxel_resolution/5),
		int(model.z_max / model.voxel_resolution))).astype('bool')
	
	# Create sdf voxel
	scene_sdf = np.ones((
		int(model.x_max / model.voxel_resolution/5),
		int(model.y_max / model.voxel_resolution/5),
		int(model.z_max / model.voxel_resolution))) * 6.0
	for sdf_step in range(0,6):
		scene_sdf[
			sdf_step : scene_sdf.shape[0]-sdf_step,
			sdf_step : scene_sdf.shape[1]-sdf_step,
			sdf_step * 5: scene_sdf.shape[2]
		] = sdf_step
	return scene_top_down, scene_occ, scene_sdf

def get_convex_hull(mat):
	# Compute the convex hull of the primary object
	array_scatter = []
	## calculate points for each contour
	for si in range(mat.shape[0]):
		for sj in range(mat.shape[1]):
			if mat[si, sj]:
				array_scatter.append([[si, sj]])
	if array_scatter == []:
		return None
	else:
		array_scatter = np.array(array_scatter).astype('int32')
		conv_hull = cv2.convexHull(array_scatter, False)
		return conv_hull

def get_objective_value(scene_occ, obj_occ, i, j, k, scene_top_down, obj_top_down, scene_sdf, choose_objective='sdf', sdf_remain_terms='1234', reorder=False):
	obj_size = obj_occ.shape
	scene_size = scene_occ.shape
	c = 1
	if choose_objective == 'sdf':
		objective_ = (
			k 
			+ 5.0/2 * (scene_sdf[i:i+obj_size[0], j:j+obj_size[1], k:k+obj_size[2]] * obj_occ).sum() / (obj_occ).sum() \
			+ 20.0/2 * (1.0 - obj_occ.sum()**(1/3) / (obj_size[0] * obj_size[1] * obj_size[2])**(1/3))
		)
		if reorder:
			objective_ += 160.0/2 * (1.0 - obj_occ.sum()**(1/3) / (scene_size[0] * scene_size[1] * scene_size[2])**(1/3))
	return objective_

def get_scene_object_heightmaps(scene_occ, obj_occ, i, j):
	obj_size = obj_occ.shape
	scene_size = scene_occ.shape
	scene_top_down = scene_size[2] - np.argmax(np.flip(scene_occ[i:i+obj_size[0], j:j+obj_size[1]], 2),2)\
		 - np.logical_not(np.any(scene_occ[i:i+obj_size[0], j:j+obj_size[1]], 2)) * scene_size[2]
	obj_top_down = obj_size[2] - np.argmax(np.flip(obj_occ, 2),2) - np.logical_not(np.any(obj_occ, 2)) * obj_size[2]
	obj_bottom_up = np.argmax(obj_occ, 2) + np.logical_not(np.any(obj_occ, 2)) * 1000 # * obj_size[2]
	return scene_top_down, obj_top_down, obj_bottom_up

def get_current_scene_state(model):
	import matplotlib.pyplot as plt
	viewMatrix = p.computeViewMatrix(cameraEyePosition=[0.5, 0, 3.1],cameraTargetPosition=[0.5, 0, 0],cameraUpVector=[1, 0, 0])
	projectionMatrix = p.computeProjectionMatrixFOV(fov=10.0,aspect=1.0,nearVal=0.1,farVal=3.2)
	width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=54, height=54,viewMatrix=viewMatrix,projectionMatrix=projectionMatrix)
	depthImg2 = depthImg[11:54-11,12:54-10]
	from PIL import Image
	import PIL
	# create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
	proj_matrix = np.asarray(projectionMatrix).reshape([4, 4], order="F")
	view_matrix = np.asarray(viewMatrix).reshape([4, 4], order="F")
	tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

	# create a grid with pixel coordinates and depth values
	y, x = np.mgrid[-1:1:2 / depthImg2.shape[0], -1:1:2 / depthImg2.shape[1]]
	y *= -1.
	x, y, z = x.reshape(-1), y.reshape(-1), depthImg2.reshape(-1)
	h = np.ones_like(z)

	pixels = np.stack([x, y, z, h], axis=1)
	# filter out "infinite" depths
	#pixels = pixels[z < 0.999]
	pixels[:, 2] = 2 * pixels[:, 2] - 1

	# turn pixels to world coordinates
	points = np.matmul(tran_pix_world, pixels.T).T
	points /= points[:, 3: 4]
	points = points[:, :3]
	scene_top_down = np.zeros(depthImg2.shape)
	for i in range(depthImg2.shape[0]):
		for j in range(depthImg2.shape[1]):
			scene_top_down[i,j] = np.round((points[depthImg2.shape[0]*i+j,2])/model.voxel_resolution*100.0)
	scene_top_down = np.asarray(Image.fromarray(scene_top_down).transpose(PIL.Image.FLIP_LEFT_RIGHT).transpose(PIL.Image.FLIP_TOP_BOTTOM))
	scene_top_down = scene_top_down.astype('int')
	return scene_top_down

def update_scene_occ(last_scene_top_down, scene_top_down, scene_occ):
	#update_region = [scene_occ.shape[0],scene_occ.shape[1],0,0]
	for i_ in range(scene_occ.shape[0]):
		for j_ in range(scene_occ.shape[1]):
			if last_scene_top_down[i_,j_] != scene_top_down[i_, j_]:
				scene_occ[i_,j_,0:scene_top_down[i_,j_]] = True
			#update_region = [min(update_region[0], i_), min(update_region[1], j_), max(update_region[2], i_), max(update_region[3], j_)]
	return scene_top_down, scene_occ

def construct_sdf(scene_occ, kernal, use_cuda):
	conv_scene_occ = np.pad(scene_occ, ((5,5),(5,5),(25,25)), 'constant', constant_values=((1,1),(1,1),(1,0)))
	k_i = k_j = kernal.shape[0]
	k_k = kernal.shape[2]
	if use_cuda:
		input_ = torch.tensor(conv_scene_occ).float().cuda()
		# First slice the horizontal planes into patches of k_i*k_i, and view the z dimention as channel (i.e., what input_.permute(2,0,1).unsqueeze(0) did)
		slices = F.unfold(input_.permute(2,0,1).unsqueeze(0), kernel_size=k_i, dilation=1, stride=1).permute(0,2,1).view(1,input_.shape[0]-k_i+1,input_.shape[1]-k_i+1,input_.shape[2],k_i,k_i)
		# Second, slice the vertical dimension
		slices = slices.reshape((input_.shape[0]-k_i+1)*(input_.shape[1]-k_i+1),input_.shape[2],k_i*k_i).unfold(dimension=1,size=k_k,step=1).reshape(input_.shape[0]-k_i+1 ,input_.shape[1]-k_i+1 ,input_.shape[2]-k_k+1,k_i,k_i,k_k)
		values = (slices.reshape(scene_occ.shape[0]*scene_occ.shape[1]*scene_occ.shape[2],k_i, k_j, k_k) * kernal).reshape(scene_occ.shape[0]*scene_occ.shape[1]*scene_occ.shape[2], k_i*k_j*k_k)
		values = (values == 0)*5.0+values
		scene_sdf = torch.min(values, dim=-1)[0].reshape(scene_occ.shape).cpu().numpy()
		torch.cuda.empty_cache()
	else:
		input_ = torch.tensor(conv_scene_occ).float()
		# First slice the horizontal planes into patches of k_i*k_i, and view the z dimention as channel (i.e., what input_.permute(2,0,1).unsqueeze(0) did)
		slices = F.unfold(input_.permute(2,0,1).unsqueeze(0), kernel_size=k_i, dilation=1, stride=1).permute(0,2,1).view(1,input_.shape[0]-k_i+1,input_.shape[1]-k_i+1,input_.shape[2],k_i,k_i)
		# Second, slice the vertical dimension
		slices = slices.reshape((input_.shape[0]-k_i+1)*(input_.shape[1]-k_i+1),input_.shape[2],k_i*k_i).unfold(dimension=1,size=k_k,step=1).reshape(input_.shape[0]-k_i+1 ,input_.shape[1]-k_i+1 ,input_.shape[2]-k_k+1,k_i,k_i,k_k)
		values = (slices.reshape(scene_occ.shape[0]*scene_occ.shape[1]*scene_occ.shape[2],k_i, k_j, k_k) * kernal).reshape(scene_occ.shape[0]*scene_occ.shape[1]*scene_occ.shape[2], k_i*k_j*k_k)
		values = (values == 0)*5.0+values
		scene_sdf = torch.min(values, dim=-1)[0].reshape(scene_occ.shape).numpy()
	return scene_sdf

def get_method_name(choose_objective):
	if choose_objective == 'sdf':
		mtd_str_ = 'SDFM'
	return mtd_str_

def pack_an_object(model, env, shape_codes, best_obj_idx, best_r_id, best_i, best_j, best_k, dict_obj_occ, list_obj_occ, packed_volumes, is_simulate=True):
	obj_occ = dict_obj_occ[str(best_obj_idx)+'_r'+str(best_r_id)]
	obj_size = obj_occ.shape
	list_obj_occ.append({'start_pos':[best_i,best_j,best_k], 'obj_occ':obj_occ, 'obj_idx':best_obj_idx})
	# 2. Place the couple/object.
	tmp_obj1 = copy.deepcopy(model.obj_list[best_obj_idx]).apply_transform(rotation_matrix(best_r_id*np.pi/4, [0, 0, 1]))
	model.obj_pos[best_obj_idx][0] = -model.x_max/2 - tmp_obj1.bounds[0][0] + best_i*model.voxel_resolution*5
	model.obj_pos[best_obj_idx][1] = -model.y_max/2 - tmp_obj1.bounds[0][1] + best_j*model.voxel_resolution*5
	model.obj_pos[best_obj_idx][2] = -model.z_max - tmp_obj1.bounds[0][2] + best_k*model.voxel_resolution
	obj_filename = os.path.join('./autostore/models/our_oriented_decomp/', np.sort(os.listdir('./autostore/models/our_oriented_dataset/'))[shape_codes[best_obj_idx]])
	obj_file_basename = os.path.basename(obj_filename)
	model.obj_rot[best_obj_idx] = best_r_id * np.pi / 4
	obs, reward, _, info = env.insert_a_packing_object(obj_file_basename[:-4], \
		([0.5+model.obj_pos[best_obj_idx][0].numpy()/100.0, model.obj_pos[best_obj_idx][1].numpy()/100.0, (model.obj_pos[best_obj_idx][2].numpy()+model.z_max)/100.0+0.02], \
		p.getQuaternionFromEuler([0,0,model.obj_rot[best_obj_idx][0].numpy().tolist()])), color=[np.random.rand(), np.random.rand(),np.random.rand(),1])#[1*(obj_idx/len(model.obj_list)), 0, 1*(1 - obj_idx/len(model.obj_list)), 0.5])
	p.changeDynamics(env.obj_ids['rigid'][-1], -1, mass=model.obj_list[best_obj_idx].volume*np.random.rand()*1)
	packed_volumes.append(obj_occ.sum()/5)#.append(model.obj_list[best_obj_idx].volume)
	if is_simulate:
		for _ in range(60):
			p.stepSimulation()
	return list_obj_occ, packed_volumes

def update_objective_maps(scene_top_down, scene_occ, scene_sdf, dict_obj_occ, obj_idx, choose_objective, require_stability, dict_objective, update_region=None, sdf_remain_terms='1234', update_local=False, reorder=False):
	# Specify the invalid values and infalid height for each method.
	best_obj_idx, best_i, best_j, best_k, best_r_id = 0, 0, 0, 0, 0
	if choose_objective == 'mta':
		invalid_value = -1
	elif choose_objective in ['dblf', 'hm', 'sdf', 'random', 'first']:
		invalid_value = 1000000.0
	invalid_height = 1000000.0

	for r in range(4):
		# Get basic size information of the bin and the object
		obj_occ = dict_obj_occ[str(obj_idx)+'_r'+str(r)]
		scene_size = scene_occ.shape
		obj_size = obj_occ.shape
		
		# If the object is too large to fit the bin at the current orientation, then skip.
		if (np.array(obj_size) > np.array(scene_size)).any():
			continue
		
		# Initialize the objective maps. 
		# Note that local update scheme is only used in ['dblf', 'mta', 'hm', 'sdf'].
		if choose_objective in ['sdf']: 
			# If the objective map exists in the dictionary load the map; otherwise, initialize the map.
			if str(obj_idx)+'_r'+str(r)+'_map' in dict_objective.keys():
				objective_map = dict_objective[str(obj_idx)+'_r'+str(r)+'_map']
				just_initialized = False
			else:
				objective_map = np.ones((scene_size[0]-obj_size[0]+1, scene_size[1]-obj_size[1]+1, 2)) * invalid_value
				objective_map[:,:,1] = invalid_height
				just_initialized = True

		# Speicify the searching region.
		if choose_objective in ['sdf']:
			# If there is not a specific update region, or the map is just initialized, update the whole map
			if just_initialized:
				update_region = [0, 0, scene_size[0], scene_size[1]]
			else:
				np_where_0, np_where_1 = np.where(scene_top_down!=dict_objective[str(obj_idx)+'_r'+str(r)+'_sceneTD'])
				update_region = [min(np_where_0), min(np_where_1), max(np_where_0), max(np_where_1)]
				#print ('update_region', update_region)
			search_i = list(range(max(0, update_region[0]-obj_size[0]+1), min(scene_size[0]-obj_size[0]+1, update_region[2]+obj_size[0]), 1))
			search_j = list(range(max(0, update_region[1]-obj_size[1]+1), min(scene_size[1]-obj_size[1]+1, update_region[3]+obj_size[1]), 1))
		
		# If stability is required to measure, compute the mass center of object
		if require_stability:
			nonzeros = np.nonzero(obj_occ)
			tmp_obj_center = np.array([np.mean(nonzeros[0]), np.mean(nonzeros[1]), np.mean(nonzeros[2])])

		for i in search_i:
			for j in search_j:
				scene_part_top_down, obj_top_down, obj_bottom_up = get_scene_object_heightmaps(scene_occ, obj_occ, i, j)
				k = max((scene_part_top_down-obj_bottom_up).reshape(-1).tolist())
				if k < 0:
					k = 0
				if k >= scene_size[2]-obj_size[2]:
					objective_map[i,j,0] = invalid_value
					objective_map[i,j,1] = invalid_height
					continue
				# 2. Stability Check
				if not require_stability:
					is_stable = True
				else:
					is_stable = False
					if k <= 5:
						support_face = (np.any(obj_occ[:,:,0:5], axis=-1))
					else:
						support_face = (np.any(np.logical_and(\
							scene_occ[i:i+obj_size[0], j:j+obj_size[1], k-6:k-1], obj_occ[:,:,0:5]), axis=-1))
					support_polygon = get_convex_hull(support_face)
					if support_polygon is not None and cv2.pointPolygonTest(support_polygon, (tmp_obj_center[0], tmp_obj_center[1]), 1)>0.5: # It IS STABLE
						is_stable = True
				if not is_stable:
					objective_map[i,j,0] = invalid_value
					objective_map[i,j,1] = invalid_height
					continue
				# 3. Find the placement with minimal objective to execute
				# Find the best object and placement with minimal objective value
				if choose_objective in ['sdf']:
					objective_map[i,j,0] = get_objective_value(scene_occ, obj_occ, i, j, k, scene_part_top_down, obj_top_down, scene_sdf, choose_objective, sdf_remain_terms=sdf_remain_terms, reorder=reorder)
					objective_map[i,j,1] = k
		dict_objective[str(obj_idx)+'_r'+str(r)+'_map'] = objective_map
		dict_objective[str(obj_idx)+'_r'+str(r)+'_sceneTD'] = scene_top_down
	return dict_objective

def sdf_placement_couple_reorder_concave_with_simulation(model, kernal, shape_codes, env, dict_obj_occ, require_stability=False, choose_objective='sdf', fix_sequence_length=5, use_cuda=False, sdf_remain_terms='1234', vol_dec=False):
	# This function implements bin packing for irregular objects with controllable arriving order.
	# given the model and the environment
	"""
	:param model: the packing model class, including information of each object, the location and orientation, bin size, etc.
	:param kernal: torch.tensor(2*5+1, 2*5+1, 2*5*5+1) [Could cuda if use_cuda=True]. For PyTorch fast computation of the SDF field.
	:param shape_codes: []. The ids for all objects to be packed.
	:param env: the PyBullet environment class, including the robot arms, the bin, the physical simulation etc.
	:param dict_obj_occ: pre-scanned object heightmaps and pre-constructed object occ.
	:param require_stability: True if only stable placements are valid. The stability is measured by if the object's mass center is located inside the support polygon corresponding to a target location.
	:param choose_objective: 'sdf', 'dblf', 'hm', 'mta', 'random' or 'first'
	:param fix_sequence_length: length of the packing buffer. Objects packing order is controllable in the buffer. Once the buffer is full, packing ends.
	:param use_cuda: set as 'True' to use CUDA computation to speed up the SDF construction.
	"""
	# Define Method Names
	mtd_str_ = get_method_name(choose_objective)
	
	# Initialize Timers
	reorder_tot_time = 0.0
	left_list = []
	list_obj_occ = []
	packed_volumes = []

	# Define packing list
	obj_idx = 0
	list_to_place = np.arange(len(model.obj_list)).astype('int').tolist()

	# Create scene voxel
	last_scene_top_down, scene_occ, scene_sdf = packing_scene_initialize(model)
	
	# Initialize a 2D objective map for each object
	dict_objective = {}
	
	# Find the best object and the best position to place
	while list_to_place != []:
		# Initialize
		success_ = False
		c = 1 # weighting for x and y location
		if choose_objective == 'mta':
			max_objective = 0.0
		elif choose_objective in ['dblf', 'hm', 'sdf', 'random', 'first']:
			min_objective = 1000000.0
		obj_start_ = time.time()

		scene_top_down = get_current_scene_state(model)
		_, scene_occ = update_scene_occ(last_scene_top_down, scene_top_down, scene_occ)
		
		searching_list = list_to_place[0:fix_sequence_length]

		if vol_dec:
			searching_list = sorted(searching_list, key=lambda coord:model.obj_bbox_volume_list[coord], reverse=True)
			
		if choose_objective == 'sdf':
			update_local=True
		else:
			update_local=False
		# Find the best object and placement with minimal (MTA: maximal) objective value
		for obj_idx in searching_list:
			if vol_dec:
				dict_objective = update_objective_maps(scene_top_down, scene_occ, scene_sdf, dict_obj_occ, obj_idx, choose_objective, require_stability, dict_objective, sdf_remain_terms=sdf_remain_terms, update_local = update_local, reorder=False)
			else:
				dict_objective = update_objective_maps(scene_top_down, scene_occ, scene_sdf, dict_obj_occ, obj_idx, choose_objective, require_stability, dict_objective, sdf_remain_terms=sdf_remain_terms, update_local = update_local, reorder=True)
			for r in range(4):
				# Only if the couple fails to get placed, decouple it, and consider placing the primary object as an individual object
				obj_occ = dict_obj_occ[str(obj_idx)+'_r'+str(r)]
				if str(obj_idx)+'_r'+str(r)+'_map' not in dict_objective:
					continue
				objective_map = dict_objective[str(obj_idx)+'_r'+str(r)+'_map']
				
				# Find the best object and placement with minimal objective value
				if choose_objective in ['sdf']:
					if np.min(objective_map[...,0]) < min_objective:
						min_objective = np.min(objective_map[...,0])
						argmin_ = np.argmin(objective_map[...,0])
						best_i = int(np.floor(argmin_/objective_map.shape[1]))
						best_j = int(argmin_%objective_map.shape[1])
						best_k = int(objective_map[best_i, best_j, 1])
						best_r_id = r
						best_obj_idx = obj_idx
						success_ = True
			if vol_dec and success_:
				break
		# If the best object and the best placement is found
		if success_:
			reorder_tot_time += time.time() - obj_start_
			# 1. Pack the object
			if vol_dec:
				print (mtd_str_+' vol-dec placement SUCCESS. Placed item', best_obj_idx, 'Best', (best_i, best_j, best_k, best_r_id))
			else:
				print (mtd_str_+' re-order placement SUCCESS. Placed item', best_obj_idx, 'Best', (best_i, best_j, best_k, best_r_id))
			list_to_place.remove(best_obj_idx)
			list_obj_occ, packed_volumes = pack_an_object(model, env, shape_codes, best_obj_idx, best_r_id, best_i, best_j, best_k, dict_obj_occ, list_obj_occ, packed_volumes)
			# 2. Update the scene occupancy and scene sdf
			#    A Local Update Scheme: update only regions around the newly placed object
			update_start_ = time.time()
			last_scene_top_down = scene_top_down
			#print ('update_region', update_region)
			if choose_objective == 'sdf':
				scene_sdf = construct_sdf(scene_occ, kernal, use_cuda)
			reorder_tot_time += time.time() - update_start_
		else:
			reorder_tot_time += time.time() - obj_start_
			decouple_start_ = time.time()
			reorder_tot_time += time.time() - decouple_start_
			# If the buffer is full and the remaining objects cannot be placed
			# Stop the packing algorithm and quit.
			if vol_dec:
				print (mtd_str_+' vol-dec placement FAIL. Unable to place items', list_to_place)
			else:
				print (mtd_str_+' re-order placement FAIL. Unable to place items', list_to_place)
			break

	print ('Stability Required:', require_stability)
	if choose_objective == 'sdf' and sdf_remain_terms != '1234':
		print ('SDF remain terms:', sdf_remain_terms)
	print ('Time for '+mtd_str_+'re-order placement:', reorder_tot_time)

	return scene_occ, scene_sdf, list_obj_occ, reorder_tot_time, packed_volumes, last_scene_top_down