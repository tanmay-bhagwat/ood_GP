import torch
from torch.distributions import Normal

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
            train_loss = model.nll()

            ln_prior = Normal(loc=-3.0, scale=0.5)
            ls_prior = Normal(loc=2, scale=0.5)
            ll_prior = Normal(loc=2.5, scale=0.5)

            reg_loss = -ln_prior.log_prob(model.log_noise) + \
                -ls_prior.log_prob(model.kernel.log_sigvar) + -ll_prior.log_prob(model.kernel.log_lengthscale).sum()
            
            train_loss += reg_loss
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
                    val_loss = model.nll()

                self.scheduler.step(val_loss)
        
        return train_loss.item(), val_loss.item()

    def train(self, epochs, data_loader, val_loader, reload_states=False):

        count_val, count_tn = 0, 0
        f = 0.1
        patience, tn_limit = 25, 20
       
        model = self.model.to(device=self.device)
        train_loss_ls, val_loss_ls = [], []
        reload_epoch = 0
        epoch = 0
        if reload_states:
            checkpoint = torch.load("best_checkpoint.pth")
            reload_epoch = checkpoint["epoch"]
        epoch += reload_epoch

        while epoch < epochs:
            train_loss, val_loss = self.train_epoch(data_loader, val_loader)
            epoch += 1
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

            # need val loss to stay delta close to train loss
            # if epoch >= 8:
            #     if abs(val_loss - train_loss) > 0.5*train_loss:
            #         print("======= Validation loss more than 50% away from train loss! =======")
            #         print("Stopping...")
            #         break
            #     if abs(val_loss - train_loss) > f*train_loss:
            #         count_val += 1
            #         if count_val == patience:
            #             print("======= Validation loss exceeded train loss! =======")
            #             print(f"Val loss exceeded train loss over {patience} epochs, calling early stop...")
            #             break
            #     else:
            #         count_val = 0

            # if val loss not decreasing over some iters, stop 
            if epoch >= 10: 
                # hacky way to bypass errors when restarting training at an epoch > 150 with no val loss history
                try:
                    if torch.abs(val_loss - val_loss_ls[-2]) <= 0.001:
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
                except:
                    pass
        
        return train_loss_ls, val_loss_ls