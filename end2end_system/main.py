import argparse
import glob
import pandas as pd
import random
import torch
import numpy as np
import math
import json
from numpy import percentile
from torch.utils.data import Dataset, DataLoader
from utils import get_specs_for_all_channels, cross_train, get_normalize_params, init_model, get_list_construction_error, z_score, roc_curve
from dataset import SoundDataset

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--train_folder", type=str)
    args_parser.add_argument("--val_folder", type=str)
    args_parser.add_argument("--test_folder", type=str)
    args_parser.add_argument("--fpr_threshold", type=float, default=0.1)
    args_parser.add_argument("--seq_len", type=int, default=10)
    args_parser.add_argument("--epochs", type=int, default=500)
    args_parser.add_argument("--batch_size", type=int, default=64)
    args_parser.add_argument("--hidden_size", type=int, default=128)
    args_parser.add_argument("--latent_size", type=int, default=64)
    args_parser.add_argument("--n_freqs", type=int, default=1025)

    
    args = args_parser.parse_args()
    metadata_df = pd.read_csv(args.metadata)
    y_train = pd.iloc[:5000]["label"].tolist()
    y_val = pd.iloc[5000:7400]["label"].tolist()
    y_test = pd.iloc[7400:]["label"].tolist()


    list_training_files = glob.glob(f"./{args.train_folder}/channel_1/*.wav")
    list_training_files.sort(key=lambda s: int(s.split('/')[-1].split('_')[-1].replace('.wav','')))
    list_specs_human, freq_bins, time_bins = get_specs_for_all_channels(f"{args.train_folder}", length=0.15, nb_files=10000, sr=16000)
    train_pairs = [(t[0], t[1], t[2]) for t in zip(list_specs_human, y_train, list_training_files)]
    train_val_pairs = train_pairs
    random.shuffle(train_val_pairs)

    list_of_mean, list_of_std = get_normalize_params(np.array([t[0] for t in train_val_pairs]))
    print(list_of_mean)
    print(list_of_std)
    # spec_transforms = transforms.Normalize(mean=list_of_mean, std=list_of_std)
    label_dict ={"presence": 1, "empty": 0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len=args.seq_len
    epochs=args.epochs
    batch_size=args.batch_size
    hidden_size=args.hidden_size
    latent_size=args.latent_size
    n_freqs=args.n_freqs
    best_model_state, histories = cross_train(n_folds=5, train_val_pairs=train_val_pairs, label_dict=label_dict, data_folder=args.train_folder, 
                                            epochs=epochs, batch_size=batch_size, hidden_size=hidden_size, latent_size=latent_size, 
                                            n_freqs=n_freqs, seq_len=seq_len, part='full', mean=list_of_mean[0], std=list_of_std[0], 
                                            transform=None, device=device, num_of_workers=0)
    # train_pairs = train_val_pairs[:4000]
    list_val_files = glob.glob(f"./{args.val_folder}/channel_1/*.wav")
    list_val_files.sort(key=lambda s: int(s.split('/')[-1].split('_')[-1].replace('.wav','')))
    list_val_specs_human, val_freq_bins, val_time_bins = get_specs_for_all_channels(f"{args.val_folder}", length=0.15, nb_files=10000, sr=16000)
    val_pairs = [(t[0], t[1], t[2]) for t in zip(list_val_specs_human, y_val, list_val_files)]
    # val_pairs = combined_pairs[5000:7500]

    restore_best_model = init_model(label_dict, input_size=n_freqs*4, hidden_size=hidden_size, latent_size=latent_size, seq_len=seq_len)
    # k=3
    restore_best_model = torch.load(f'full_best_model_LSTM_AE_h_{hidden_size}_l{latent_size}_f{n_freqs}.pt')
    # restore_best_model.load_state_dict(best_model_state)
    restore_best_model.to(torch.device('cpu'))
    

    val_ds = SoundDataset(train_val_pairs, label_dict, folder='.', part='full', mean=0, std=0, transform=None)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle = False,
                            drop_last = False)
    criterion = torch.nn.MSELoss(reduction="none")
    list_train_losses, list_train_label, list_train_names, list_train_enc_vec, list_train_dec_specs = get_list_construction_error(train_val_pairs, best_model=restore_best_model, criterion=criterion, part='all', mean=list_of_mean[0], std=list_of_std[0], transform=None, bs=64, label_dict=label_dict)
    list_val_losses, list_val_label, list_val_names, list_val_enc_vec, list_val_dec_specs = get_list_construction_error(val_pairs, best_model=restore_best_model, criterion=criterion, part='all', mean=list_of_mean[0], std=list_of_std[0], transform=None, bs=64, label_dict=label_dict)
    train_losses_mean = np.mean(list_train_losses)
    print("Mean", train_losses_mean)
    train_losses_var = np.var(list_train_losses)
    print("Var", train_losses_var)
    val_losses_mean = np.mean(list_val_losses)
    print("Mean", val_losses_mean)
    val_losses_var = np.var(list_val_losses)
    print("Var", val_losses_var)
    q25, q75 = percentile(list_train_losses, 25), percentile(list_train_losses, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    outliers = [x for x in list_train_losses if x < lower or x > upper]
    filtered_list_train_losses = [x for x in list_train_losses if x >= lower and x <= upper]

    mean = np.mean(filtered_list_train_losses)
    std = np.std(filtered_list_train_losses)
    list_z_score_train = [z_score(t, mean, std) for t in list_train_losses]
    list_z_score_val = [z_score(t, mean, std) for t in list_val_losses]
    thresholds = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    list_fpr, list_tpr = roc_curve(y_true=np.array(y_val), y_prob=np.array(list_z_score_val), thresholds=thresholds)
    fpr_threshold = args.fpr_threshold
    best_threshold = thresholds[np.where(np.array(list_fpr) < fpr_threshold)[0][0]]
    params_dict = {
        "seq_len":seq_len,
        "epochs":epochs,
        "batch_size":batch_size,
        "hidden_size":hidden_size,
        "latent_size":latent_size,
        "n_freqs":n_freqs,
        "best_threshold": best_threshold
    }
    json.dump(params_dict, "model_params.json")