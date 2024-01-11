import torch

class Train():
    def __init__(self, writer) -> None:
        self.writer = writer

    def train_model(self, model, train_loader, val_loader, epochs, optimizer, criterion):
        for epoch in range(1, epochs + 1):
            model.train() 
            train_loss = 0.0
            
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0) 

            train_loss = train_loss / len(train_loader.dataset)
            self.writer.add_scalar("Loss/train", loss, epoch)
            
            print(f"Epoch {epoch}/{epochs}, Training Loss: {train_loss:.4f}")
            
            model.eval()  
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item() * data.size(0)
                    
                    _, predicted = output.max(1)
                    correct += (predicted == target).sum().item()
                    total += target.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            self.writer.add_scalar("Loss/valid", val_loss, epoch)
            val_accuracy = correct / total * 100
            self.writer.add_scalar("Acc/valid", val_accuracy, epoch)
            
            print(f"Epoch {epoch}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
