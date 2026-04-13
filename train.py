import torch


class GPTrainer:
    def __init__(self, model, optimizer, device="cpu") -> None:
        self.model = model
        self.optimizer = optimizer 
        self.device = device       

    def train_epoch(self, data_loader, val_loader):
        model = self.model
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, mode="min", patience=1, threshold=1e-2)

        model.train()
        
        # One batch is the whole dataset
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            model.fit(data, labels)
            train_loss = model.mll()
            print(f"Batch {batch_idx}, batch loss: {train_loss:.3f} ")

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        print(f"Total loss: {train_loss:.3f}")

        if val_loader is not None:
            model.eval()
            
            val_loss = 0
            with torch.no_grad():
                for _, (val_data, val_labels) in enumerate(val_loader):
                    val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
                    model.fit(val_data, val_labels)
                    val_loss = model.mll()

                scheduler.step(val_loss)
        
        return train_loss, val_loss

    def train(self, epochs, data_loader, val_loader):

        count_val, count_tn = 0, 0
        delta = 0.01
        patience, tn_limit = 3, 5
       
        model = self.model.to(device=self.device)
        train_loss_ls, val_loss_ls = [], []
        for epoch in range(epochs):
            train_loss, val_loss = self.train_epoch(data_loader, val_loader)
            train_loss_ls.append(train_loss.item())
            val_loss_ls.append(val_loss.item())
            print(f"Epoch {epoch}: Train loss = {train_loss:.3f}, val loss = {val_loss:.3f}")

            if val_loss <= min(val_loss_ls):
                print("Saving best model...\n")
                kernel = self.model.kernel
                torch.save(model.state_dict(), 'best_model.pth')
                torch.save(kernel.state_dict(), 'best_kernel.pth')
            
            if val_loss > train_loss + delta:
                count_val += 1
                if count_val == patience:
                    print(f"Val loss exceeded train loss over {patience} epochs, calling early stop...")
                    break

            if epoch >= 1:
                if train_loss > train_loss_ls[-2]:
                    count_tn += 1
                    print("======= WARNING: Training loss increased! =======")
                    if count_tn == tn_limit:
                        print(f"Train loss increased for {tn_limit} consecutive epochs, stopping...")
                        break
                else:
                    count_tn = 0 
        
        return train_loss_ls, val_loss_ls