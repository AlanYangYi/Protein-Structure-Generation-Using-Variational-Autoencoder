import torch
from torch.utils.data import TensorDataset,random_split,Subset
import Model as MODEL
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
import loss_functions as lf
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.rms import RMSD




def process_xyz_file_optimized(file_path):
    frames = []
    atom_count = None

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                atom_count = int(line.strip())  # 获取每个帧的原子数
                continue
            if i % (atom_count + 2) < 2:  # 跳过每帧的前两行
                continue
            split_line = line.split()
            if len(split_line) == 4:  # 确保行有4个元素
                frames.append([float(split_line[1]), float(split_line[2]), float(split_line[3])])

    # 转换为NumPy数组，然后转换为tensor
    frames_array = np.array(frames, dtype=np.float32).reshape(-1, atom_count, 3)
    tensor = torch.tensor(frames_array)

    # 标准化

    min_val, max_val = tensor.min(), tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    torch.save({
        'tensor': normalized_tensor,
        'min_val': min_val,
        'max_val': max_val
    }, 'saved_tensor.pt')

    return normalized_tensor, min_val, max_val




# Process aligned.xyz file
# aligned_xyz_path = 'aligned_2pd7_unbound.xyz'
# xyz_tensor,min_val,max_val = process_xyz_file_optimized(aligned_xyz_path)
def load_tensor_from_file(tensor_file_path):
    data = torch.load(tensor_file_path)
    tensor = data['tensor']
    min_val = data['min_val']
    max_val = data['max_val']
    # 如果需要，也可以重新进行标准化或其他处理
    return tensor, min_val, max_val


tensor_file_path='saved_tensor.pt'
xyz_tensor, min_val, max_val = load_tensor_from_file(tensor_file_path)

print(xyz_tensor.shape)


dataset = TensorDataset(xyz_tensor)
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])







#
BATCH_SIZE = 10
#
# test_dataset=Subset(dataset, range(train_size,len(dataset)))
train_dataset=Subset(dataset, range(0,3000))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
sample_loader=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Return the sizes of the train and test datasets, for confirmation
print(f"train_size:{len(train_dataset)}, test_size:{len(test_dataset)}")



no_cuda = False
cuda_available = not no_cuda and torch.cuda.is_available()


EPOCH = 40
SEED = 3407
ITERATION =0

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if cuda_available else "cpu")
print(device)
model = MODEL.VAE(device).to(device)



def loss_function(recon_x, x, mu, logvar,w_recon,w_kl,w_angle=10):
    #BCE = F.binary_cross_entropy(recon_x, x,reduction='sum')
    # mse=F.mse_loss(recon_x,x,reduction='mean')
    #mse=lf.weight_mse_loss(recon_x,x,BATCH_SIZE)
    mse=lf.reconstruction_loss(recon_x,x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # angle_loss=calculate_Ramachandran_angle.compute_ramachandran_loss(x, recon_x, calculate_Ramachandran_angle.n_indices,calculate_Ramachandran_angle.ca_indices, calculate_Ramachandran_angle.c_indices)
    angle_loss=0
    return w_recon*mse + w_kl*KLD+w_angle*angle_loss, mse, KLD


def adjust_weights(iteration):
    # 根据迭代次数设置wRecon的值
    if iteration <= 6000:  # 0.3M
        wRecon = 10.0
    elif iteration <= 20000:  # 1.0M
        wRecon = 1.0

    # 根据迭代次数设置wKL的值
    if iteration <= 4000:  # 50K
        wKL = 1e-5
    elif iteration <= 6000:  # 100K
        wKL = 5e-5
    elif iteration <= 7000:  # 150K
        wKL = 1e-4
    elif iteration <= 4000:  # 200K
        wKL = 5e-4
    elif iteration <= 5000:  # 250K
        wKL = 0.001
    elif iteration <= 10000:  # 500K
        wKL = 0.005
    else:  # 超过500K
        wKL = 0.01

    return wRecon, wKL

optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0005)
#scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
def train(epoch):
    global ITERATION
    # Sets the module in training mode.
    model.train()
    train_loss = 0
    train_mse=0
    train_kld=0
    train_rmsd=0
    train_angle=0
    z_values=[]
    #     batch_idx, (data, label) =enumerate(train_loader).__next__()
    for batch_idx, (data,) in enumerate(train_loader):
        # w_recon, w_kl = adjust_weights(ITERATION)
        data = data.view(-1, 3549)  # Flatten the data
        data = data.to(device)  #
        recon_batch, mu, logvar,z = model(data)
        z_values.append((z.detach().cpu().numpy()))
        loss,mse,kld = loss_function(recon_batch, data, mu, logvar,w_recon=10,w_kl=0.0001)

        recon_data=(data*(max_val - min_val))+min_val
        recon_coor=(recon_batch*(max_val - min_val))+min_val
        recon_rmsd=torch.sqrt(F.mse_loss(recon_data,recon_coor))
        # angle_loss=calculate_Ramachandran_angle.compute_ramachandran_loss(recon_data,recon_coor,calculate_Ramachandran_angle.n_indices,calculate_Ramachandran_angle.ca_indices, calculate_Ramachandran_angle.c_indices)

        optimizer.zero_grad()
        loss.backward()
        ITERATION = ITERATION + 1
        optimizer.step()

        train_loss += loss.item()
        train_mse+=mse.item()
        train_kld+=kld.item()
        train_rmsd+=recon_rmsd.item()
        # train_angle+=angle_loss.item()



    # Process each tuple in the list


    #scheduler.step(train_rmsd)
    # print(f" w_r:{w_recon}   w_k: {w_kl}")
    # print(f" Epoch: {epoch}  Iteration: {ITERATION}     train_loss: {train_loss / len(train_loader)} ")
    print((f"Epoch: {epoch}   Iteration: {ITERATION}    train_cons_rmsd: {train_rmsd / len(train_loader)}"))
    print((f"Epoch: {epoch}   Iteration: {ITERATION}    train_mse: {train_mse / len(train_loader)}"))
    print((f"Epoch: {epoch}   Iteration: {ITERATION}     train_kld: {train_kld / len(train_loader.dataset)}"))
    # print((f"Epoch: {epoch}   Iteration: {ITERATION}     train_angle: {train_angle / len(train_loader.dataset)}"))
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
    #           ' -->grad_value:', torch.mean(parms.grad))
    if epoch==EPOCH:
        s=input('save_model_parameters? y/n')
        if s=='y':
            torch.save(model.state_dict(), 'model_with_distance_loss.pth')
        return np.concatenate(z_values, axis=0)
    else:
        return None


    #return np.concatenate(z_values, axis=0)


def test(epoch):
    # Sets the module in evaluation mode
    model.eval()
    test_loss = 0
    test_rmsd=0
    z_values = []
    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            data = data.view(-1, 3549)  # Flatten the data
            data = data.to(device)
            recon_batch, mu, logvar,z = model(data)
            z_values.append((z.detach().cpu().numpy()))
            recon_data = (data * (max_val - min_val)) + min_val
            recon_coor = (recon_batch * (max_val - min_val)) + min_val
            recon_rmsd = torch.sqrt(F.mse_loss(recon_data, recon_coor))
            test_loss += loss_function(recon_batch, data, mu, logvar,w_recon=2,w_kl=0.001)[0].item()
            test_rmsd+=recon_rmsd

    test_loss /= len(test_loader.dataset)
    #print(f"Epoch: {epoch}    test_loss: {test_loss} ")
    print((f" Epoch: {epoch}  test_cons_rmsd: {test_rmsd/len(test_loader)}"))
    print("=================================================================")






def sample(sample_class):
    model.eval()
    with (torch.no_grad()):
        sample = torch.randn(500, 120).to(device) #64 is hidden dimension
        sample = model.decode(sample).cpu()
        sample=(sample*(max_val - min_val))+min_val
        sample=sample.reshape(sample.shape[0],1183,3)
        sample=sample.numpy()
        sample=np.concatenate(sample, axis=0)
    return np.savetxt(f'generate_conformations_randomly_{sample_class}.txt', sample, fmt='%.4f')




def generate_conformations(SEED,T):

    for seed in SEED:
        model.set_temperature(T)
        generated_conformations = []
        embedding_points = []
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model.eval()


        with torch.no_grad():
            for (data,) in train_loader:
                data = data.view(-1, 3549)  # Flatten the data
                data = data.to(device)
                recon_batch, mu, logvar, z = model(data)
                recon_coor = (recon_batch * (max_val - min_val)) + min_val
                batch_conformations = recon_coor.reshape(-1, 1183,3).cpu().numpy()
                generated_conformations.append(batch_conformations)
                embedding_points.append(z.cpu().numpy())
        generated_conformations = np.concatenate(generated_conformations, axis=0)
        generated_conformations = np.concatenate(generated_conformations, axis=0)
        generated_conformations = np.around(generated_conformations, 4)
        embedding_points = np.concatenate(embedding_points, axis=0)
        np.save(f'sample_points_{T}_{seed}.npy', embedding_points)
        np.save(f'generating_conformation_{T}_{seed}.npy', generated_conformations)
    return f'generating_conformation_{T}_{seed}.npy'

    #np.savetxt(f'generating_conformation_{T}.txt', generated_conformations)
#
for epoch in range(1,EPOCH+1):
    z1=train(epoch)
    # test(epoch)
    epoch+=1
    # if epoch+1 == EPOCH:
    #     np.save('training_embedding_points.npy', z1)
    #     np.save('test_embedding_points_without_normalized.npy')



model.load_state_dict(torch.load('model.pth'))

def replace_coordinates(pdb_lines, new_coordinates, frame_number):
    """
    Replace the coordinates in the PDB lines with new coordinates.

    :param pdb_lines: List of lines from the original PDB file.
    :param new_coordinates: List of new coordinates to replace in the PDB file.
    :return: List of modified PDB lines with updated coordinates.
    """
    updated_lines = []
    coord_idx = 0  # Index for the new_coordinates list

    for line in pdb_lines:
        if line.startswith("TITLE"):
            # Updating the TITLE line with the current frame number
            updated_lines.append(f"MODEL     {frame_number}\n")
        else:
            if line.startswith("ATOM"):
                # Extracting the non-coordinate parts of the line
                line_start = line[:30]  # Everything before the coordinates
                line_end = line[54:]  # Everything after the coordinates

                # Replacing the coordinates
                x, y, z = map(float, new_coordinates[coord_idx])
                new_line = f"{line_start}{x:8.3f}{y:8.3f}{z:8.3f}{line_end}"
                updated_lines.append(new_line)

                coord_idx += 1
            else:
                if line.startswith("END"):
                    updated_lines.append("ENDMDL\n")
                else:
                    updated_lines.append(line)


    return updated_lines

def visualize_conformations(filename):
    sample_1_lines = np.load(filename)

    frame_length = 1183
    sample_1_frames = [sample_1_lines[i:i + frame_length] for i in range(0, len(sample_1_lines), frame_length)]

    num_frames = len(sample_1_frames)
    print(num_frames)

    with open('pdb_temp.pdb', 'r') as file:
        pdb_temp_lines = file.readlines()

    # Processing each frame and writing to the new PDB file
    with open(f'sample_{filename[:-4]}.pdb', 'w') as new_file:
        frame_number = 0
        for frame in sample_1_frames:
            updated_pdb_lines = replace_coordinates(pdb_temp_lines, frame, frame_number)
            new_file.writelines(updated_pdb_lines)
            frame_number += 1

    ref = mda.Universe('2pd7.pdb')
    ref = ref.select_atoms('not (resid 36) and (segid A) and protein and (not name H*) ')

    traj = mda.Universe(f'sample_{filename[:-4]}.pdb')

    rmsd_analysis = RMSD(traj, reference=ref, select='backbone and resid 38-184 and (not resid 108-110)')
    rmsd_analysis.run()
    rmsd_results = rmsd_analysis.rmsd
    np.savetxt(f'rmsd_results_{filename[:-4]}.txt', rmsd_results, fmt='%.2f')

input('continue')
filename=generate_conformations([1],1)
visualize_conformations(filename)
filename=generate_conformations([1],5)
visualize_conformations(filename)
filename=generate_conformations([1],10)
visualize_conformations(filename)


