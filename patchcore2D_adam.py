from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization import ImageVisualizer
from anomalib.metrics import AUROC, F1Score, Evaluator
from anomalib.data import Folder
import os
import json
from datetime import datetime 

ignore_categories = [
    "baseclampbody",
    "baseclampbolt",
    "gripper_open",
    "gripper_closed",
    "t2m1bd",
    "t2m1bi",
    "t3m1c",
    "t4m1e",
]

def run_anomalib_on_dataset(dataset_base_dir: str, log_file_path: str):
    """
    Iterates through all categories in a dataset and runs the anomalib
    pipeline (training and testing) for each category, logging results to a file.

    Args:
        dataset_base_dir (str): The path to the base directory containing
                                the category subdirectories.
        log_file_path (str): The path to the output log file.
    """
    # Get a list of all category directories
    categories = [
        d for d in os.listdir(dataset_base_dir) 
        if os.path.isdir(os.path.join(dataset_base_dir, d))
    ]

    # Initialize a list to hold all results
    all_results = []

    # Iterate through each category
    for category in categories:
        if category in ignore_categories:
            continue
        print(f"--- Processing category: {category} ---")
        
        # Construct the full path to the current category
        category_path = os.path.join(dataset_base_dir, category)
        
        # Initialize the datamodule with the new root path
        datamodule = Folder(
            name=category,
            root=category_path,
            normal_dir="train",
            abnormal_dir="test",
            mask_dir="ground_truth",
            train_batch_size=8, 
            eval_batch_size=8,
            num_workers=4
        )

        datamodule.setup()

        # Create metrics
        metrics = [
            AUROC(fields=["pred_score", "gt_label"]),
           # F1Score(fields=["pred_label", "gt_label"])
        ]

        # Create evaluator with metrics
        #evaluator = Evaluator(test_metrics=[])

        
        visualizer = ImageVisualizer(
            fields_config={"anomaly_map": {"normalize": True}}
        )
        model = Patchcore(visualizer=visualizer)

        engine = Engine(
            max_epochs=10,
            )
        
        # Run the training and testing for the current category
        engine.fit(datamodule=datamodule, model=model)
        #predictions = engine.predict(model, datamodule=datamodule)  
        test_results = engine.test(model=model, datamodule=datamodule)
        
        print(f"Test results for {category}: {test_results}")
        #print(f"Prediction results for {category}: {predictions}")

        #watcher = weightwatcher.WeightWatcher(model=model)
        #details = watcher.analyze()
        #summary = watcher.get_summary(details) 
#
        #summary_path = category + "_watcher_log.txt"
        #with open(summary_path, 'w') as file:
        #    file.write(summary)

        # Add the results to our list
        all_results.append({
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "results": test_results
        })


    # Write all results to the log file as JSON
    with open(log_file_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll categories processed. Results written to {log_file_path}")


if __name__ == '__main__':
    dataset_base_dir = "datasets/adam3d_unsupervised_cushion5pc"
    log_file_path = "patchcore_2D_test_results_adam_masked_c05.json"
    run_anomalib_on_dataset(dataset_base_dir, log_file_path)
    