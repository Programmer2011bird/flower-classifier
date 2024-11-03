import torchvision.models as models
from torch.nn import Linear


class_names = [
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

NUM_CLASSES:int = 102

MODEL:  models.ResNet = models.resnet18(pretrained=True)
MODEL.fc = Linear(MODEL.fc.in_features, NUM_CLASSES)

