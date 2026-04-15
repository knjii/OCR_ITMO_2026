import matplotlib.pyplot as plt
import torch
import tqdm
import os
import json

# Train function on one epoch
def train_epoch(model, train_loader, criterion, optimizer, metrics):
    
    """
    Input:
        model (nn.Module class) - using model
        train_loader (torch.Dataloader) - DataLoader for train dataset
        criterion (function) - using criterion in training
        optimizer (function) - using optimier in training
        metrics (function) - calculated metrics for researching
    Output:
        avg_loss (float) - average loss
        avg_metrics (dict) - average metrics

    """
    device='cuda' if torch.cuda.is_available() else 'cpu'

    model.train()

    running_loss = 0.0
    running_metrics = {}
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training")
        
    for images, texts in pbar:

        images = images.to(device)
        texts = texts.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, texts)
        loss.backward()
        optimizer.step()

        batch_metrics = metrics(outputs, texts)
        for k, v in batch_metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v
        
        # train_metrics = metrics(outputs, texts)
        running_loss += loss.item()
        n_batches += 1
    
    if n_batches == 0:
        return 0.0, {}
    
    avg_loss = running_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}

    return avg_loss, avg_metrics


# Validate function on one epoch
def validate_epoch(model, val_loader, criterion, metrics):
    """
    Input:
        model (nn.Module class) - using model
        val_loader (torch.Dataloader) - DataLoader for val dataset
        criterion (function) - using criterion in validation
        metrics (function) - calculated metrics for researching
    Output:
        avg_loss (float) - average loss
        avg_metrics (dict) - average metrics

    """
    device='cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()

    running_loss = 0.0
    running_metrics = {}
    n_batches = 0

    pbar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for images, texts in pbar:
            images = images.to(device)
            texts = texts.to(device)

            outputs = model(images)

            loss = criterion(outputs, texts)
            
            batch_metrics = metrics(outputs, texts)
            for k, v in batch_metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v

            # val_metrics = metrics(outputs, texts)
            running_loss += loss.item()
            
            n_batches += 1

    if n_batches == 0:
        return 0.0, {}

    avg_loss = running_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}

    return avg_loss, avg_metrics


# Counting parameters of model function
def count_parameters(model):
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return parameters


# Train function
def train_model(model, num_epochs,
                train_loader, val_loader,
                criterion, optimizer, scheduler, metrics,
                objective_metric=None):
    """
    Input:
        model (nn.Module class) - using model
        train_loader (torch.Dataloader) - DataLoader for train dataset
        val_loader (torch.Dataloader) - DataLoader for val dataset
        criterion (function) - using criterion
        optimizer (function) - using optimizer
        scheduler (function) - using sheduler
        metrics (function) - calculated metrics for researching
        objective_metric (string) - objective metric for comparising a models
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    best_val = 0.0

    if os.path.exists('best_model.pth'):

        checkpoint = torch.load('best_model.pth', map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['best_val']
        history = checkpoint['history']

        print(f"Загружено: эпоха {start_epoch}, Best Val: {best_val}")

    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, metrics
        )

        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, metrics
        )

        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_metrics[objective_metric])
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        if epoch == 0:
            metrics_names = train_metrics.keys()
            history.update({f'train_{k}': [] for k in metrics_names})
            history.update({f'val_{k}': [] for k in metrics_names})
        for k in train_metrics.keys():
            history[f'train_{k}'].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])

        if objective_metric == 'loss':
            objective = val_loss
        else:
            objective = val_metrics[objective_metric]

        if objective > best_val:
            best_val = objective
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val': best_val,
                'history': history
            }, 'best_model.pth')
            print(f"  Сохранена лучшая модель! Best Val: {best_val:.4f}")

    number_of_parameters = count_parameters(model) / 1e6
    final_results = {
        'Number of parameters': number_of_parameters
    }
    final_results.update({k: v[-1] for k, v in history.items()})

    # сохранение метрик
    with open('final_metrics.json','w') as f:
        json.dump(final_results, f, indent=2)


def plot_metrics(history):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Losses')

    plt.subplot(1, 2, 2)
    for metric in history.keys():
        if metric.startswith('train_') and 'loss' not in metric:
            plt.plot(history[metric], label=metric)
    plt.legend()
    plt.title('Metrics')

    plt.show()

