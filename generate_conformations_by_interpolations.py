import torch
from torch.utils.data import TensorDataset
import CVAE_model as MODEL
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader

def load_tensor_from_file(tensor_file_path):
    data = torch.load(tensor_file_path)
    tensor = data['tensor']
    min_val = data['min_val']
    max_val = data['max_val']
    # 如果需要，也可以重新进行标准化或其他处理
    return tensor, min_val, max_val

tensor_file_path='saved_tensor.pt'
xyz_tensor, min_val, max_val = load_tensor_from_file(tensor_file_path)

def gen_conformations_with_equally_spaced_interpolations(select_num_class):
    sample_step=70
    points_small = torch.linspace(-40, 40, sample_step)
    x_small, y_small = torch.meshgrid(points_small, points_small)
    x_flat_small = x_small.flatten()
    y_flat_small = y_small.flatten()
    data_points_small = torch.stack((x_flat_small, y_flat_small), dim=1)
    label=torch.full((sample_step*sample_step,), select_num_class)
    np.save(f'sample_points_by_interpolation_class_{select_num_class}.npy',data_points_small.numpy())

    return TensorDataset(data_points_small, label)

def sample_conforamtions(TensorDataset,model,seed,select_num_class):
    generated_conformations = []
    loader=DataLoader(TensorDataset, batch_size=50, shuffle=False)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model.eval()

    with torch.no_grad():
        for (data, label) in loader:
            data = data.to(device)
            label = label.to(device)
            recon_batch= model.decode(data, label)
            recon_coor = (recon_batch * (max_val - min_val)) + min_val
            batch_conformations = recon_coor.reshape(-1, 1183, 3).cpu().numpy()
            generated_conformations.append(batch_conformations)


    generated_conformations = np.concatenate(generated_conformations, axis=0)
    generated_conformations = np.around(generated_conformations, 4)
    generated_conformations = np.concatenate(generated_conformations, axis=0)
    np.savetxt(f'generating_conformation_interpolations_{select_num_class}.txt', generated_conformations,fmt='%.4f')

no_cuda = False
cuda_available = not no_cuda and torch.cuda.is_available()
SEED = 3407

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if cuda_available else "cpu")
print(device)
model = MODEL.CVAE(3,device).to(device)
model.load_state_dict(torch.load('model_2.pth'))


select_number_class=1
tensor_data=gen_conformations_with_equally_spaced_interpolations(select_number_class)
sample_conforamtions(tensor_data,model,SEED,select_number_class)

