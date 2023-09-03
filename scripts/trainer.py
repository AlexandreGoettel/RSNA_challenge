"""Implement class for general DL training and monitoring."""
from tqdm import tqdm, trange
import torch
import numpy as np
from matplotlib import pyplot as plt


class Trainer:
    """Train networks, monitor train/val loss and accuracy, plot progress."""

    def __init__(self, model, loss, train_loader, val_loader,
                 optimiser, scheduler, max_epoch, device="cuda"):
        """
        Set variables for training.

        Most vars are self explanatory.
        loss must return a tuple (total_loss, losses) where total_loss is a float
            and losses is a dict (can be of length 1). This will be useful for plots.
        """
        self.model = model                # The neural network model to be trained
        self.loss = loss                  # The loss function to optimise
        self.train_loader = train_loader  # DataLoader for the training set
        self.val_loader = val_loader      # DataLoader for the validation set
        self.optimiser = optimiser        # The optimisation algorithm
        self.scheduler = scheduler        # Learning rate scheduler
        self.max_epoch = max_epoch        # Maximum number of training epochs
        self.device = device              # Where to train the model (cpu/gpu)


    def train(self, verbose=True):
        """Implement torch training loop."""
        # Book-keeping
        train_loss, val_loss = np.zeros(len(self.train_loader)), np.zeros(len(self.val_loader))
        book = np.zeros((self.max_epoch, 2, 3))
        self.model.to(self.device)
        for epoch in trange(self.max_epoch, desc="Training"):
            # Switch to training mode
            self.model.train()
            for ibatch, (data, truth) in tqdm(enumerate(self.train_loader),
                                              desc=f"Epoch {epoch+1}", position=1,
                                              leave=False, total=len(self.train_loader)):
                data, truth = data.to(self.device), truth.to(self.device)
                self.optimiser.zero_grad()
                # data = utils.batch_normalise(data)  # TODO
                output = self.model(data)
                loss, single_losses = self.loss(output, truth)

                # Update weights
                loss.backward()
                self.optimiser.step()

                # Book-keeping
                # output = output.detach().numpy()
                # train_accuracy[:, ibatch] = utils.getAccuracy(output,  # TODO
                #                                               truth.numpy())
                train_loss[ibatch] = loss.item()

            # book[epoch, 0, :2] = utils.weighedAverage(*train_accuracy)
            book[epoch, 0, 2] = np.mean(train_loss)
            tqdm.write(f"[TRAIN] loss: {book[epoch, 0, 2]:.4e}")

            # Switch to validation mode
            self.model.eval()
            with torch.no_grad():
                for ibatch, (data, truth) in tqdm(enumerate(self.val_loader),
                                                  desc=f"Epoch {epoch+1} validating", position=1,
                                                  leave=False, total=len(self.val_loader)):
                    data, truth = data.to(self.device), truth.to(self.device)
                    # data = batch_normalise(data)  # TODO
                    output = self.model(data)
                    loss, single_losses = self.loss(output, truth)

                    # Book-keeping
                    # output = output.detach().numpy()
                    # val_accuracy[:, ibatch] = utils.getAccuracy(output, truth.numpy())
                val_loss[ibatch] = loss.item()
            # book[epoch, 1, :2] = utils.weighedAverage(*val_accuracy)
            book[epoch, 1, 2] = np.mean(val_loss)
            tqdm.write(f"[VALID] loss: {book[epoch, 1, 2]:.4e}")

            # Update scheduler
            self.scheduler.step(book[epoch, 1, 2])

            # Save model progress
            # torch.save(model.state_dict(), model_path)
            # np.save("book.npy", book)

            # Plot training progress
            if verbose and epoch >= 2:
                self.do_plots(book, epoch)

    def do_plots(self, book, epoch):
        """Plot training and validation losses and accuracies."""
        # TODO: plot accuracies
        # TODO: plot single loss terms
        plt.figure()
        ax = plt.subplot(111)
        ax.set_yscale("log")
        ax.plot(book[:epoch+1, 0, 2], label="Training")
        ax.plot(book[:epoch+1, 1, 2], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total loss")
        ax.legend(loc="upper right")
        plt.savefig("loss.png")
        plt.close()
