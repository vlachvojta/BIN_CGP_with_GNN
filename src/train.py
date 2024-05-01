
import torch
import torch_geometric


def main():
    train()


def train():
    num_epochs = 100

    for epoch in range(1, num_epochs):
        model.train()
        out = model(data)
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            out = model(data)
            loss = criterion(out, data.y)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


if __name__ == '__main__':
    main()
