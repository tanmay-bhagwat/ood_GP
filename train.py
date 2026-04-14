import torch

class GPTrainer:
    def __init__(self, model, optimizer, scheduler, device="cpu"):
        self.model = model
        self.optimizer = optimizer 
        self.device = device
        self.scheduler = scheduler       

    def train_epoch(self, data_loader, val_loader):
        model = self.model

        model.train()
        
        # One batch is the whole dataset
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            model.fit(data, labels)
            train_loss = model.mll()
            # print(f"Batch {batch_idx}, batch loss: {train_loss:.3f} ")

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            #model.kernel.log_lengthscale.data.clamp_(min=-1.0, max=1.0)
            
        # print(f"Total loss: {train_loss:.3f}")

        if val_loader is not None:
            model.eval()

            val_loss = 0
            with torch.no_grad():
                for _, (val_data, val_labels) in enumerate(val_loader):
                    val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
                    model.fit(val_data, val_labels)
                    val_loss = model.mll()

                self.scheduler.step(val_loss)
        
        return train_loss.item(), val_loss.item()

    def train(self, epochs, data_loader, val_loader):

        count_val, count_tn = 0, 0
        delta = 0.01
        patience, tn_limit = 3, 20
       
        model = self.model.to(device=self.device)
        train_loss_ls, val_loss_ls = [], []
        for epoch in range(epochs):
            train_loss, val_loss = self.train_epoch(data_loader, val_loader)
            train_loss_ls.append(train_loss)
            val_loss_ls.append(val_loss)
            print(f"Epoch {epoch}: Train loss = {train_loss:.3f}, val loss = {val_loss:.3f}")

            if val_loss <= min(val_loss_ls):
                print("Saving best model...\n")
                # print(model.alpha.shape, model.L.shape)
                checkpoint = {"epoch":epoch, "model_state_dict": self.model.state_dict(), 
                              "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(),
                              "loss":val_loss}
                torch.save(checkpoint, "best_checkpoint.pth")

            if val_loss > train_loss + delta:
                count_val += 1
                if count_val == patience:
                    print("======= Validation loss exceeded train loss! =======")
                    print(f"Val loss exceeded train loss over {patience} epochs, calling early stop...")
                    break

            if epoch >= 1:
                if val_loss > val_loss_ls[-2]:
                    count_tn += 1
                    print("======= Validation loss increased =======")
                    if count_tn == tn_limit:
                        print(f"Validation loss increased for {tn_limit} consecutive epochs, stopping...")
                        checkpoint = {"epoch":epoch, "model_state_dict": self.model.state_dict(), 
                            "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(),
                            "loss":val_loss}
                        torch.save(checkpoint, "last_checkpoint.pth")
                        break
                else:
                    count_tn = 0 
        
        return train_loss_ls, val_loss_ls