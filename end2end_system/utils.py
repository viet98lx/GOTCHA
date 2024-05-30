import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import LSTMAE
from tqdm import tqdm, tqdm_notebook
import glob
import re
import librosa
from datetime import datetime
from dataset import SoundDataset

def get_specs(list_audio, sr=16000):
  list_specs = []
  for audio in list_audio:
    # Play audio file
    y, sr = librosa.load(audio, sr=sr)
    D = librosa.stft(y, n_fft=2048, hop_length=256, win_length=1024)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    list_specs.append(S_db)
  return list_specs

def get_spec_for_one_audio(audio, length, sr=16000):
    y, sr = librosa.load(audio, sr=sr)
    y = y[:int(sr*length)]
    D = librosa.stft(y, n_fft=2048, hop_length=256, win_length=1024)  # STFT of y
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
    time_bins = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db, freq_bins, time_bins

def get_specs_for_all_channels(scenery_pattern, length, nb_files, sr=16000):
  list_specs = []
  freq_bins = []
  time_bins = []
  missing_idx = []
  # list_audio_scene_1.sort()
  # print(len(list_audio_scene_1))  
  for i in range((nb_files)):
    if len(glob.glob(f"{scenery_pattern}/channel_1/{i}.wav")) > 0:
      spec_channel_1, freq_bins, time_bins = get_spec_for_one_audio(audio=f"{scenery_pattern}/channel_1/{i}.wav", length=length, sr=sr)
      print(spec_channel_1.shape)
      spec_channel_2, _, _ = get_spec_for_one_audio(audio=f"{scenery_pattern}/channel_2/{i}.wav", length=length, sr=sr)
      spec_channel_3, _, _ = get_spec_for_one_audio(audio=f"{scenery_pattern}/channel_3/{i}.wav", length=length, sr=sr)
      spec_channel_4, _, _ = get_spec_for_one_audio(audio=f"{scenery_pattern}/channel_4/{i}.wav", length=length, sr=sr)
      combined_specs = np.concatenate((spec_channel_1, spec_channel_2, spec_channel_3, spec_channel_4), axis=0)
      # print(combined_specs.shape)
      list_specs.append(combined_specs)
    else:
      print("Missing specs idx: ", i)
      missing_idx.append(i)
  return list_specs, freq_bins, time_bins, missing_idx

def init_model(label_dict, input_size, hidden_size, latent_size, seq_len):
    # determine if the system supports CUDA
    if torch.cuda.is_available():
      device = torch.device("cuda:0")
    else:
      device = torch.device("cpu")

    # init model
    net = LSTMAE(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size, dropout_ratio=0.2, seq_len=seq_len).to(device)
    # print(net)

    return net


def get_normalize_params(inputs_set):
    list_of_mean = []
    list_of_std = []
    list_of_mean.append(inputs_set[:,:,:].mean())
    list_of_std.append(inputs_set[:,:,:].std())
    return list_of_mean, list_of_std


def train_model(criterion, epoch, model, optimizer, train_iter, batch_size, clip_val, device, log_interval, scheduler=None):
    """
    Function to run training epoch
    :param criterion: loss function to use
    :param epoch: current epoch index
    :param model: pytorch model object
    :param model_type: model type (only ae/ ae+clf), used to know if needs to calculate accuracy
    :param optimizer: optimizer to use
    :param train_iter: train dataloader
    :param batch_size: size of batch (for logging)
    :param clip_val: gradient clipping value
    :param log_interval: interval to log progress
    :param scheduler: learning rate scheduler, optional.
    :return mean train loss (and accuracy if in clf mode)
    """
    model_type = "LSTMAE"
    model.train()
    loss_sum = 0
    pred_loss_sum = 0
    correct_sum = 0
    torch.manual_seed(0)

    num_samples_iter = 0
    # pbar = tqdm(total=len(train_iter.dataset))
    for batch_idx, data in tqdm(enumerate(train_iter, 1), position=0, leave=True, desc="Training Step"):
        if len(data) == 3:
            x, labels, filename = data['spectrogram'].to(device), data['label'].to(device), data['filename']
            x = torch.permute(x, (0, 2, 1))
            h0 = torch.zeros(1, x.size()[0], model.encoder.hidden_size)
            c0 = torch.zeros(1, x.size()[0], model.encoder.hidden_size)
        else:
            x = data.to(device)
        # Zero gradients
        optimizer.zero_grad()

        num_samples_iter += x.size()[0]  # Count number of samples seen in epoch (used for later statistics)

        # model.encoder.lstm_enc.flatten_parameters()
        # Forward pass & loss calculation
        # print("Input")
        # print(x[0])
        # print(x[10])
        model_enc_out, model_dec_out = model(x)
        # print("encoded out")
        # print(model_enc_out)
        if model_type == 'LSTMAE_CLF':
            # For MNIST classifier
            model_out, out_labels = model_out
            pred = out_labels.max(1, keepdim=True)[1]
            correct_sum += pred.eq(labels.view_as(pred)).sum().item()
            # Calculate loss
            mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
            loss = mse_loss + ce_loss
        elif model_type == 'LSTMAE_PRED':
            # For S&P prediction
            model_out, preds = model_out
            labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
            preds = preds[:, :-1]  # Take preds up to T-1
            mse_rec, mse_pred = criterion(model_out, data, preds, labels)
            loss = mse_rec + mse_pred
            pred_loss_sum += mse_pred.item()
        else:
            # Calculate loss
            loss = criterion(model_dec_out, x)

        # Backward pass
        loss.backward()
        print(f"Loss at batch {batch_idx}: {loss.detach().item()}")
        loss_sum += loss.item()

        # Gradient clipping
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        # Update model params
        optimizer.step()
        # pbar.update(1)

    train_loss = loss_sum / (len(train_iter.dataset)/batch_size)
    print('Train Epoch: {} [{} samples]\tLoss: {:.8f}'.format(
                epoch, len(train_iter.dataset), train_loss))
    train_pred_loss = pred_loss_sum / (len(train_iter.dataset)/batch_size)
    train_acc = round(correct_sum / (len(train_iter.dataset)/batch_size) * 100, 2)
    acc_out_str = f'; Average Accuracy: {train_acc}' if model_type == 'LSTMAECLF' else ''
    print(f'Train Average Loss: {train_loss}{acc_out_str}')

    return train_loss, train_acc, train_pred_loss


def eval_model(criterion, model, val_iter, batch_size, device, mode='Validation'):
    """
    Function to run validation on given model
    :param criterion: loss function
    :param model: pytorch model object
    :param model_type: model type (only ae/ ae+clf), used to know if needs to calculate accuracy
    :param val_iter: validation dataloader
    :param mode: mode: 'Validation' or 'Test' - depends on the dataloader given.Used for logging
    :return mean validation loss (and accuracy if in clf mode)
    """
    # Validation loop
    model_type = "LSTMAE"
    model.eval()
    loss_sum = 0
    correct_sum = 0
    with torch.no_grad():
        for data in tqdm(val_iter, position=0, leave=True, desc="Validation Step"):
            if len(data) == 3:
                x, labels, filename = data['spectrogram'].to(device), data['label'].to(device), data['filename']
                x = torch.permute(x, (0, 2, 1))
                h0 = torch.zeros(1, x.size()[0], model.encoder.hidden_size)
                c0 = torch.zeros(1, x.size()[0], model.encoder.hidden_size)
            else:
                x = data.to(device)
            # model.encoder.lstm_enc.flatten_parameters()
            # print("Input val")
            # print(x[0])
            # print(x[10])
            model_enc_out, model_dec_out = model(x)
            # print(model_dec_out)
            if model_type == 'LSTMAE_CLF':
                model_out, out_labels = model_out
                pred = out_labels.max(1, keepdim=True)[1]
                correct_sum += pred.eq(labels.view_as(pred)).sum().item()
                # Calculate loss
                mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
                loss = mse_loss + ce_loss
            elif model_type == 'LSTMAE_PRED':
                # For S&P prediction
                model_out, preds = model_out
                labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
                preds = preds[:, :-1]  # Take preds up to T-1
                mse_rec, mse_pred = criterion(model_out, data, preds, labels)
                loss = mse_rec + mse_pred
            else:
                # Calculate loss for none clf models
                loss = criterion(model_dec_out, x)

            loss_sum += loss.item()
    val_loss = loss_sum / (len(val_iter.dataset)/batch_size)
    val_acc = round(correct_sum / len(val_iter.dataset) * 100, 2)
    acc_out_str = f'; Average Accuracy: {val_acc}' if model_type == 'LSTMAECLF' else ''
    print(f' {mode}: Average Loss: {val_loss}{acc_out_str}')
    return val_loss, val_acc

def inference_construction_error(best_model, data, criterion):
    # criterion = torch.nn.MSELoss(reduction='mean')
    best_model.eval()
    with torch.no_grad():
        enc_out, dec_out = best_model(data)
        loss = criterion(dec_out, data)
    return loss.detach().numpy(), enc_out.detach().numpy(), dec_out.detach().numpy()

def cross_train(n_folds, train_val_pairs, label_dict, data_folder, epochs=100, batch_size=32, hidden_size=128, latent_size=16, n_freqs=257, seq_len=1, part='all', mean=0, std=0, transform=None, device="cpu", num_of_workers=0):

    # normalize the data
    # train_df, test_df = normalize_data(train_df, test_df)
    # n_folds = 5
    total_samples = len(train_val_pairs)
    samples_per_part = int(total_samples/n_folds)
    list_history = []
    best_val_loss = np.inf
    for k in range(n_folds):
        history = {'train_loss':[],  'val_loss':[]}
        train_pairs = train_val_pairs[0:k*samples_per_part]+train_val_pairs[(k+1)*samples_per_part:]
        val_pairs = train_val_pairs[k*samples_per_part:(k+1)*samples_per_part]
        # print("Len of train: ", len(train_pairs))
        # print("Len of val: ", len(val_pairs))
        # init train data loader
        # print(transform)
        train_ds = SoundDataset(train_pairs, label_dict, data_folder, part, mean, std, transform=transform)
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  shuffle = True,
                                  drop_last = True)
        val_ds = SoundDataset(val_pairs, label_dict, data_folder, part, mean, std, transform=transform)
        val_loader = DataLoader(val_ds,
                                batch_size=batch_size,
                                shuffle = False,
                                drop_last = False)
    
        # init model
        model = init_model(label_dict, input_size=n_freqs*4, hidden_size=hidden_size, latent_size=latent_size, seq_len=seq_len)
    
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=0.1, eps=1e-07, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # train the model
        start_time = datetime.now()
        best_val_loss = np.inf
    
        for epoch in range(epochs):
            # Train loop
            train_loss, train_acc, _ = train_model(criterion, epoch, model, optimizer, train_loader, batch_size, clip_val=None, device=device, log_interval=10)
            history['train_loss'].append(train_loss)
            # train_acc_lst.append(train_acc)
            # if (epoch + 1) % (int(epochs / 10)) == 0:
                # plot_images(model, val_iter, description=args.model_type, num_to_plot=3)
            # Evaluate
            val_loss, val_acc = eval_model(criterion, model, val_loader, batch_size, device, mode='Validation')
            # LR scheduler step
            if scheduler is not None:
                scheduler.step()
            # val_acc_lst.append(val_acc)
            history['val_loss'].append(val_loss)
            if val_loss < best_val_loss:
              best_model_state = model.state_dict()
              history['best_model_state_dict'] = best_model_state
              torch.save(best_model_state, f'state_LSTM_AE_h_{hidden_size}_l_{latent_size}_f{n_freqs}.pt')
              torch.save(model, f'full_best_model_LSTM_AE_h_{hidden_size}_l{latent_size}_f{n_freqs}.pt')
              best_val_loss = val_loss
    
        end_time = datetime.now() - start_time
        print("\nTraining completed in time: {}".format(end_time))
        list_history.append(history)
    return best_model_state, list_history

def get_list_construction_error(data_pairs, best_model, criterion, part='all', mean=0, std=0, transform=None, bs=32, label_dict):
    data_ds = SoundDataset(data_pairs, label_dict, folder='.', part=part, mean=mean, std=std, transform=transform)
    data_loader = DataLoader(data_ds,
                            batch_size=bs,
                            shuffle = False,
                            drop_last = False)
    list_losses = []
    list_enc_vec = []
    list_dec_specs = []
    list_labels = []
    list_names = []
    if torch.cuda.is_available():
      device = torch.device("cuda:0")
    else:
      device = torch.device("cpu")
    for data in data_loader:
        x, label, name = data['spectrogram'], data['label'], data['filename']
        x_permuted = torch.permute(x, (0, 2, 1))
        construction_err, enc_vec, dec_spec = inference_construction_error(best_model, x_permuted, criterion)
        # print(construction_err.shape)
        list_construction_err = construction_err.mean(axis=(1, 2)).tolist()
        for idx in range(len(list_construction_err)):
            list_losses.append(list_construction_err[idx])
            list_enc_vec.append(enc_vec[idx,:])
            list_dec_specs.append(dec_spec[idx,:])
            list_labels.append(label[idx])
            list_names.append(name[idx])
    return list_losses, list_labels, list_names, list_enc_vec, list_dec_specs

def z_score(point, mean, std):
    return abs((point-mean)/std)

def roc_curve(y_true, y_prob, thresholds):
    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return (fpr, tpr)