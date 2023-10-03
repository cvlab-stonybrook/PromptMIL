

def get_default_classes():
    CLASSES = [0, 1]
    CLASS_NAMES = ['0', '1']
    return CLASSES, CLASS_NAMES


def get_tcga_brca_classes():
    CLASSES = [0, 1]
    CLASS_NAMES = ['IDC', 'ILC']
    return CLASSES, CLASS_NAMES


def get_bright_classes(num_classes):
    if num_classes == 6:
        CLASSES = [0, 1, 2, 3, 4, 5]
        CLASS_NAMES = ['0', '1', '2', '3', '4', '5']
        return CLASSES, CLASS_NAMES
    else:
        CLASSES = [0, 1, 2,]
        CLASS_NAMES = ['0', '1', '2']
        return CLASSES, CLASS_NAMES


def get_tcga_crc_classes(type):
    if type == "msi":
        CLASSES = [0, 1]
        CLASS_NAMES = ['0', '1']
    elif type == "cings":
        CLASSES = [0, 1]
        CLASS_NAMES = ['0', '1']
    elif type == "hypermutated":
        CLASSES = [0, 1]
        CLASS_NAMES = ['0', '1']
    else:
        raise NotImplementedError
    return CLASSES, CLASS_NAMES

def get_class_names(dataset_name):
    if dataset_name == 'tcga-brca':
        classes_names = get_tcga_brca_classes()
    elif dataset_name == 'bright-6':
        classes_names = get_bright_classes(6)
    elif dataset_name == 'bright-3':
        classes_names = get_bright_classes(3)
    elif dataset_name == 'tcga-crc-msi':
        classes_names = get_tcga_crc_classes("msi")
    elif dataset_name == 'tcga-crc-cings':
        classes_names = get_tcga_crc_classes("cings")
    elif dataset_name == 'tcga-crc-hypermutated':
        classes_names = get_tcga_crc_classes("hypermutated")
    elif dataset_name is None:
        print("Not specify dataset, use default dataset with label 0, 1 instead.")
        classes_names = get_default_classes()
    else:
        raise NotImplementedError
    return classes_names