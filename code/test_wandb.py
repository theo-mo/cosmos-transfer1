import wandb

# ① 初始化项目
wandb.init(project="test_docker", config={
    "epochs": 5,
    "batch_size": 64,
    "lr": 1e-3,
})

config = wandb.config

wandb.log({"loss": 'train_loss', "accuracy": 'acc', "epoch": 'epoch'})
