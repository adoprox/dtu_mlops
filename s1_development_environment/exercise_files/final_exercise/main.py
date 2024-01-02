import click
import torch
from model import MyAwesomeModel
from data import mnist
from torch import optim
from torch import nn


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    epochs = 10
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            optimizer.zero_grad()
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            output = model(images)
            # TODO: Training pass
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(train_set)}")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)
    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
