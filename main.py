from torch import argmax, load, optim, inference_mode, save, device
from torch.nn import CrossEntropyLoss, Linear
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import train_test
import dataloader


NUM_CLASSES: int = 102
EPOCHS: int = 15
LEARNING_RATE: float = 0.001
CLASS_NAMES: list[str] = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "wild geranium", "tiger lily", "moon orchid", "bird of paradise", "monkshood", 
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", 
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", 
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", 
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", 
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", 
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", 
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower", 
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia", 
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan", 
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", 
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple", 
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani", 
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", 
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "pink quill", 
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", 
    "blanket flower", "trumpet creeper", "blackberry lily", "common tulip", "wild rose"
]

MODEL: models.ResNet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
MODEL.fc = Linear(MODEL.fc.in_features, NUM_CLASSES)

TRAIN_LOADER, TEST_LOADER = dataloader.get_data()

# LOSS_FN: CrossEntropyLoss = CrossEntropyLoss()
# OPTIMIZER: optim.Adam = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

# for epoch in tqdm(range(EPOCHS)):
    # train_loss: int = train_test.train_step(LOSS_FN, OPTIMIZER, MODEL, TRAIN_LOADER)
    # test_loss: int = train_test.test_step(LOSS_FN, MODEL, TEST_LOADER)
    # 
    # save(MODEL.state_dict(), "model.pth")
 
    # print(f"TRAIN LOSS: {train_loss}")
    # print(f"TRAIN LOSS: {test_loss}")

train_test.Visualize_test_Model(MODEL, TEST_LOADER, CLASS_NAMES)

