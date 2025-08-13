from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder3D
from adam_3d_datamodule import ADAM3D
from anomalib.data import MVTec3D
import os
import json
from datetime import datetime

def run_anomalib(dataset_base_dir: str, log_file_path: str):

    #Get category list
    categories = [
        d for d in os.listdir(dataset_base_dir)
        if os.path.isdir(os.path.join(dataset_base_dir, d))
    ]

    #Initialise list to hold results
    all_results = []

    for category in categories:
        print(f"--- Processing category: {category} ---")
        
        # Construct the full path to the current category
        category_path = os.path.join(dataset_base_dir, category)

        #datamodule = MVTec3D(
        #root=dataset_base_dir,
        #category=category,
        #train_batch_size=12,
        #eval_batch_size=12,
        #num_workers=3,
        #)

        #datamodule = Folder3D(
        #    name="3d-adam",
        #    root=category_path,
        #    normal_dir="train\\good\\rgb",
        #    normal_depth_dir="train\\good\\xyz",
        #    abnormal_dir="test",
        #    abnormal_depth_dir="test",  # Will auto-detect all abnormal directories
        #    train_batch_size=8,
        #    eval_batch_size=8,
        #    num_workers=2,
        #)

        datamodule = ADAM3D(
            root=dataset_base_dir,
            category=category,
            train_batch_size=8,
            eval_batch_size=8,
            num_workers=2,
        )

        model = Patchcore()

        engine = Engine(
            #max_epochs = 10
        )

        engine.fit(datamodule=datamodule, model=model)
        test_results = engine.test(model=model, datamodule=datamodule)

        print(f"Test results for {category}: {test_results}")

         # Add the results to our list
        all_results.append({
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "results": test_results
        })

    with open(log_file_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nAll categories processed. Results written to {log_file_path}")

if __name__ == '__main__':
    dataset_base_dir = "D:\\Data\\3d-adam-tests\\3d-adam-nano-unsupervised"  # Adjust this path as needed
    log_file_path= "scores\\patchcore_adam3D_test_results.json"
    run_anomalib(dataset_base_dir, log_file_path)