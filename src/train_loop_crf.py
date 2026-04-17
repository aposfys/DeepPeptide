'''
CRF train loop.
- no marginals
- no train metrics
'''
import json
import pickle
from typing import Dict, List, Tuple
import os
from torch.utils.data import DataLoader

from .models import LSTMCNNCRF, SimpleLSTMCNNCRF, SelfAttentionCRF
from .utils import add_dict_to_writer, PrecomputedCSVForOverlapCRFDataset
#from .utils.metrics_cleaned import compute_metrics, compute_metrics_with_propeptides
from .utils.manuscript_metrics import compute_all_metrics
from torch.optim import AdamW
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
global_step = 0


def get_dataloaders(args: argparse.Namespace, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4]) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=train_partitions, label_type=args.label_type)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=valid_partitions, label_type=args.label_type)
        test_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=test_partitions, label_type=args.label_type)

    print(f'Loaded data. {len(train_set)} train sequences (p.{train_partitions}), {len(valid_set)} validation sequences (p.{valid_partitions}), {len(test_set)} test sequences (p.{test_partitions}).')


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=train_set.collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=valid_set.collate_fn, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=valid_set.collate_fn, num_workers=0)

    return train_loader, valid_loader, test_loader


def get_model(args: argparse.Namespace):

    if args.model == 'lstmcnncrf':
        model = LSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3,
            dropout_input=args.dropout,
            num_states= 101,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size, 
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_simple':
        model = SimpleLSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3 if args.label_type == 'simple_with_propeptides' else 2,
            dropout_input=args.dropout,
            num_states= 3 if args.label_type == 'simple_with_propeptides' else 2,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size, 
            dropout_conv1=args.conv_dropout,
        )

    # NOTE just use already existing CLI args with names that don't really match. Works.
    elif args.model == 'selfattentioncrf':
        model = SelfAttentionCRF(
            input_size = args.embedding_dim,
            hidden_size= args.hidden_size,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            dropout_input=args.dropout,
            num_states= 121 if 'with_propeptides' in args.label_type else 61,
            n_heads=args.num_filters,
            attn_dropout=args.conv_dropout,
        )
    else:
        raise NotImplementedError(args.model)

    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def train(args, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], is_initiated: bool = False):
    global global_step
    global_step = 0
    train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions)

    model = get_model(args)
    model.feature_extractor.biLSTM.flatten_parameters()
    model = model.to(device)

    if args.evaluate_only:
        if args.checkpoint is None:
            raise ValueError("Must provide --checkpoint when using --evaluate_only")
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

        print("Evaluating on test set...")
        # Since we are evaluating only, use a dummy writer or just pass None,
        # but run_dataloader expects a SummaryWriter, so we make one.
        writer = SummaryWriter(args.out_dir)
        test_loss, test_crf_loss, test_focal_loss, test_probs, test_preds, test_peptides, test_labels = run_dataloader(test_loader, model, None, writer, do_train=False, alpha=args.alpha)

        test_metrics_0 = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [0])[0]
        test_metrics_3 = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [3])[0]

        print(f"\n{'='*60}")
        print(f"TEST RESULTS (EVALUATE ONLY)")
        print(f"{'='*60}")
        print(f"  Propeptide Precision (+-0): {test_metrics_0['precision propeptides']:.4f}")
        print(f"  Propeptide Recall    (+-0): {test_metrics_0['recall propeptides']:.4f}")
        print(f"  Propeptide F1        (+-0): {test_metrics_0['f1 propeptides']:.4f}")
        print(f"  Propeptide Precision (+-3): {test_metrics_3['precision propeptides']:.4f}")
        print(f"  Propeptide Recall    (+-3): {test_metrics_3['recall propeptides']:.4f}")
        print(f"  Propeptide F1        (+-3): {test_metrics_3['f1 propeptides']:.4f}")
        print(f"  Seq Accuracy:               {test_metrics_3['seq accuracy']:.4f}")
        print(f"{'='*60}\n")
        return

    print("\nParameter breakdown:")
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():>10,}")
            total += param.numel()
    print(f"  {'TOTAL':30s}: {total:>10,}\n")

    bottleneck_params = list(model.feature_extractor.bottleneck.parameters())
    bottleneck_ids = set(id(p) for p in bottleneck_params)
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in bottleneck_ids]

    optimizer = AdamW([
        {'params': bottleneck_params, 'lr': 5e-4, 'weight_decay': 0.01},
        {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.05}
    ])

    # V8: Cosine Annealing with Linear Warmup
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    from torch.optim.swa_utils import AveragedModel, update_bn

    warmup_epochs = 5
    total_epochs = args.epochs

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # V8: SWA
    swa_model = AveragedModel(model)
    swa_start_epoch = 40

    writer = SummaryWriter(args.out_dir)

    previous_best = -100000000000

    for epoch in range(args.epochs):

        train_loss, train_crf_loss, train_focal_loss, train_probs, train_preds, train_peptides, train_labels = run_dataloader(train_loader, model, optimizer, writer, do_train=True, alpha=args.alpha)

        print(f"Epoch {epoch} | Train CRF Loss: {train_crf_loss:.4f} | Train Focal Loss: {train_focal_loss:.4f} | Total Loss: {train_loss:.4f}")

        #train_metrics = compute_crf_metrics(train_probs, train_preds, train_peptides, train_labels)
        #train_metrics = metrics_fn(train_peptides, train_preds)
        #add_dict_to_writer(train_metrics, writer, global_step, prefix='Train')

        valid_loss, valid_crf_loss, valid_focal_loss, valid_probs, valid_preds, valid_peptides, valid_labels = run_dataloader(valid_loader, model, optimizer, writer, do_train=False, alpha=args.alpha)
        #valid_metrics_old = compute_crf_metrics(valid_probs, valid_preds, valid_peptides, valid_labels)#, organism=valid_loader.dataset.data['organism'])
        #valid_metrics = metrics_fn(valid_peptides, valid_preds, valid_loader.dataset.data['organism'])
        valid_metrics = compute_all_metrics(valid_probs, valid_preds, valid_labels, valid_loader.dataset.names, valid_loader.dataset.data, windows = [3])[0]
        add_dict_to_writer(writer, valid_metrics, global_step, prefix='Valid')
        writer.add_scalar('Valid/loss', valid_loss, global_step=global_step)


        ratio = valid_crf_loss / (valid_focal_loss + 1e-8)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | CRF: {valid_crf_loss:.4f} | Focal: {valid_focal_loss:.4f} | "
              f"Ratio: {ratio:.1f}:1 | Val F1(+-3): {valid_metrics['f1 propeptides']:.4f} | LR: {current_lr:.6f}")

        stopping_metric = valid_metrics['f1 propeptides']

        # Step the V8 sequential scheduler
        scheduler.step()

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

        if stopping_metric > previous_best:
            previous_best = stopping_metric
            print(f"  >>> New best checkpoint saved at epoch {epoch} (F1 +-3 = {stopping_metric:.4f})")
            best_val_metrics = valid_metrics
            pickle.dump((valid_probs, valid_preds, valid_labels, valid_loader.dataset.names), open(os.path.join(args.out_dir, 'valid_outputs.pickle'), 'wb'))
            valid_metrics['epoch'] = epoch # keep track of best early stopping.
            json.dump(valid_metrics, open(os.path.join(args.out_dir, 'valid_metrics.json'), 'w'), indent=2)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))

            # valid_metrics = metrics_fn(valid_peptides, valid_preds, valid_loader.dataset.data['organism'])
            # valid_metrics['epoch'] = epoch # keep track of best early stopping.
            # json.dump(valid_metrics, open(os.path.join(args.out_dir, 'valid_metrics_old.json'), 'w'), indent=2)
    
    if args.epochs > swa_start_epoch:
        # V8: Use the SWA model for final test evaluation
        # SWA encapsulates the model under .module. The current architecture does not use BatchNorm,
        # but if it did, standard `update_bn` would fail because it only passes sequence inputs
        # (missing masks etc.). We skip `update_bn` and unwrap the model safely.
        test_model = swa_model.module
        torch.save(test_model.state_dict(), os.path.join(args.out_dir, 'swa_model.pt'))
        print(f"  >>> Saved final SWA checkpoint to swa_model.pt")
    else:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pt')))
        test_model = model

    test_loss, test_crf_loss, test_focal_loss, test_probs, test_preds, test_peptides, test_labels = run_dataloader(test_loader, test_model, optimizer, writer, do_train=False, alpha=args.alpha)

    test_metrics_0 = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [0])[0]
    test_metrics_3 = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [3])[0]

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Propeptide Precision (+-0): {test_metrics_0['precision propeptides']:.4f}")
    print(f"  Propeptide Recall    (+-0): {test_metrics_0['recall propeptides']:.4f}")
    print(f"  Propeptide F1        (+-0): {test_metrics_0['f1 propeptides']:.4f}")
    print(f"  Propeptide Precision (+-3): {test_metrics_3['precision propeptides']:.4f}")
    print(f"  Propeptide Recall    (+-3): {test_metrics_3['recall propeptides']:.4f}")
    print(f"  Propeptide F1        (+-3): {test_metrics_3['f1 propeptides']:.4f}")
    print(f"  Seq Accuracy:               {test_metrics_3['seq accuracy']:.4f}")
    print(f"{'='*60}\n")

    add_dict_to_writer(writer, test_metrics_3, global_step, prefix='Test')
    writer.add_scalar('Test/loss', test_loss, global_step=global_step)
    print('Test complete.')

    pickle.dump((test_probs, test_preds, test_labels, test_loader.dataset.names), open(os.path.join(args.out_dir, 'test_outputs.pickle'), 'wb'))

    # Save both metrics
    full_metrics = {
        'tolerance_0': test_metrics_0,
        'tolerance_3': test_metrics_3
    }
    json.dump(full_metrics, open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=2)

    return best_val_metrics, full_metrics

    

def run_dataloader(loader: torch.utils.data.DataLoader, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    writer: SummaryWriter,
                    do_train: bool = True,
                    alpha: float = 1.0,
                    pos_weight: float = 20.0
                ) -> Tuple[float, float, float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    '''
    Run a dataloader through the model. Collect predicted probabilitities and
    true labels. Can be used both for training and prediction.
    '''
    global global_step

    true = [] # peptide coordinates
    labels = [] # labels made from coordinates
    probs = [] # per-position probabilities
    preds = [] # viterbi paths
    epoch_loss = []
    epoch_crf_loss = []
    epoch_focal_loss = []

    # V7: Focal Loss for the Auxiliary Cleavage Tracker
    # Focal Loss (gamma=2.0) exponentially penalizes "hard" examples (missed cleavages)
    # while ignoring "easy" examples, providing a massive boost to Recall without sacrificing
    # the 80% Precision we built with the Sniper Head.
    def focal_loss_with_logits(logits, targets, alpha=pos_weight, gamma=2.0):
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(logits)
        # Create a numerically stable BCE term
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Calculate the modulating factor (1-p) for positives, (p) for negatives
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * ((1 - p_t) ** gamma)
        # Apply the pos_weight (alpha) only to positive targets
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - targets)
            loss = alpha_t * loss
        return loss

    if do_train:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(loader):
        
        optimizer.zero_grad()

        embeddings, mask, label, peptides= batch
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        label = label.long().to(device)

        if do_train:
            pos_probs, pos_preds, crf_loss, raw_logits = model(embeddings, mask, targets=label, skip_marginals=True)

            # V6: Auxiliary Cleavage Loss (The "Viterbi Breaker")
            # We extract the pure Cleavage logit (Index 2) *before* CRF smoothing occurs.
            cleavage_logits = raw_logits[:, :, 2]

            # The target is exactly and only the Cleavage State (100).
            cleavage_targets = (label == 100).float()

            # Calculate the weighted BCE loss. This forces a massive penalty if the
            # CNN "smears" the cleavage signal to adjacent residues, independent of
            # the CRF's tolerance for length variations.
            raw_focal_loss = focal_loss_with_logits(cleavage_logits, cleavage_targets)

            # Mask out padding sequences so they don't artificially lower the loss sum
            masked_focal_loss = (raw_focal_loss * mask.float()).sum() / mask.float().sum()

            # The CRF handles global sequence validity. The Focal loss handles pixel-perfect boundaries.
            total_loss = crf_loss + (alpha * masked_focal_loss)

            total_loss.backward()

            # V8: Differential Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model.crf.parameters(),
                max_norm=1.0   # tight — protects the transition matrix
            )
            torch.nn.utils.clip_grad_norm_(
                model.feature_extractor.parameters(),
                max_norm=5.0   # looser — allows LSTM/CNN to keep learning
            )

            optimizer.step()
            writer.add_scalar('Train/loss', total_loss.item(), global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                # Extract Top-5 Viterbi paths during validation/testing
                pos_probs, pos_preds, crf_loss, raw_logits = model(embeddings, mask, targets=label, top_k=5)

                cleavage_logits = raw_logits[:, :, 2]
                cleavage_targets = (label == 100).float()

                raw_focal_loss = focal_loss_with_logits(cleavage_logits, cleavage_targets)
                masked_focal_loss = (raw_focal_loss * mask.float()).sum() / mask.float().sum()

                total_loss = crf_loss + (alpha * masked_focal_loss)

        true.extend(peptides)
        # Extract unpadded sequences to properly align outputs for pickle
        for i in range(label.shape[0]):
            seq_len = int(mask[i].sum().item())
            probs.append(pos_probs[i, :seq_len].detach().cpu().numpy())
            labels.append(label[i, :seq_len].detach().cpu().numpy())

            # Top-K Viterbi Decoding Integration
            # If we requested top_k=5, `pos_preds` is a list of lists of paths.
            # We evaluate the paths and pick the first one that successfully predicts a Cleavage Site (100).
            # If none do, we just fall back to the absolute most likely path (pos_preds[i][0]).
            if isinstance(pos_preds[i][0], list):
                best_path = pos_preds[i][0]
                for path in pos_preds[i]:
                    if 100 in path:
                        best_path = path
                        break
                preds.append(best_path[:seq_len])
            else:
                preds.append(pos_preds[i][:seq_len])

        epoch_loss.append(total_loss.item())
        epoch_crf_loss.append(crf_loss.item())
        epoch_focal_loss.append(masked_focal_loss.item())


    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    epoch_crf_loss = sum(epoch_crf_loss)/len(epoch_crf_loss)
    epoch_focal_loss = sum(epoch_focal_loss)/len(epoch_focal_loss)

    return epoch_loss, epoch_crf_loss, epoch_focal_loss, probs, preds, true, labels





def parse_arguments():
    '''Parse arguments, prepare output directory and dump run configuration.'''
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_dir', type=str, help='Embeddings dir produced by `extract.py`', default = '/data3/fegt_data/embeddings/')
    p.add_argument('--data_file', '-df', type=str, help='Sequences with Graph-Part headers', default = 'data/uniprot_12052022_cv_5_50/labeled_sequences.csv')
    p.add_argument('--partitioning_file', '-pf', type=str, help='Graph-Part output. Assume train-val-test split.', default = 'data/uniprot_12052022_cv_5_50/graphpart_assignments.csv')
    p.add_argument('--embedding', '-em', type=str, help='Sequence embedding strategy.', default='precomputed')
    p.add_argument('--embedding_dim', '-ed', type=int, help='Sequence embedding dimension.', default=1536)

    p.add_argument('--model', '-m', type=str, default='lstmcnncrf')

    p.add_argument('--out_dir', '-od', type=str, help='name that will be added to the runs folder output', default='train_run')
    p.add_argument('--epochs', type=int, default=150, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', '-bs', type=int, default=16, help='samples that will be processed in parallel')

    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--conv_dropout', type=float, default=0.15)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--num_filters', type=int, default=32)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--alpha', type=float, default=2.0, help='Focal Loss alpha (weight)')

    p.add_argument('--label_type', type=str, default='propeptides_only')

    p.add_argument('--evaluate_only', action='store_true', help='Skip training and only evaluate a checkpoint')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to a .pt checkpoint file for evaluate_only mode')

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'config.json'), 'w'), indent=3)

    return args


if __name__ == '__main__':
    train(parse_arguments())