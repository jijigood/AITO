# F1 Distribution Repair Report

Base: `submission_600_diag_lora_boundary_resubmit.csv`

## Phase1-scaled F1 Targets
- add_neighbor: 44.3
- check_server: 44.2
- decrease_a3: 42.7
- insufficient_data: 46.6
- lift_tilt: 46.9
- pdcch_2sym: 44.3

## Check Server Probe
- check_server candidate present: 146
- phase1-like high-confidence check_server cases: 0
- candidate by current action: {'insufficient_data': 45, 'pdcch_2sym': 26, 'lift_tilt': 57, 'adjust_azimuth': 6, 'add_neighbor': 5, 'decrease_a3': 7}

## Variants
### knn_all
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_knn_all.csv`
- changed rows vs base: 101
- F1/F1_5BS distribution: {'decrease_a3': 61, 'insufficient_data': 64, 'add_neighbor': 57, 'pdcch_2sym': 30, 'lift_tilt': 47, 'adjust_azimuth': 6, 'check_server': 4}
- top transitions: ['adjust_azimuth->add_neighbor:39', 'lift_tilt->insufficient_data:28', 'adjust_azimuth->decrease_a3:15', 'insufficient_data->lift_tilt:6', 'decrease_a3->add_neighbor:4', 'decrease_a3->pdcch_2sym:3', 'insufficient_data->check_server:2', 'lift_tilt->check_server:1', 'lift_tilt->add_neighbor:1', 'decrease_a3->check_server:1', 'insufficient_data->pdcch_2sym:1']

### knn_conf40
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_knn_conf40.csv`
- changed rows vs base: 91
- F1/F1_5BS distribution: {'decrease_a3': 62, 'insufficient_data': 58, 'add_neighbor': 55, 'pdcch_2sym': 30, 'lift_tilt': 55, 'adjust_azimuth': 7, 'check_server': 2}
- top transitions: ['adjust_azimuth->add_neighbor:38', 'lift_tilt->insufficient_data:22', 'adjust_azimuth->decrease_a3:15', 'insufficient_data->lift_tilt:6', 'decrease_a3->add_neighbor:4', 'decrease_a3->pdcch_2sym:3', 'insufficient_data->check_server:2', 'insufficient_data->pdcch_2sym:1']

### knn_conf50
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_knn_conf50.csv`
- changed rows vs base: 84
- F1/F1_5BS distribution: {'decrease_a3': 61, 'insufficient_data': 54, 'add_neighbor': 53, 'lift_tilt': 59, 'pdcch_2sym': 30, 'adjust_azimuth': 10, 'check_server': 2}
- top transitions: ['adjust_azimuth->add_neighbor:36', 'lift_tilt->insufficient_data:18', 'adjust_azimuth->decrease_a3:14', 'insufficient_data->lift_tilt:6', 'decrease_a3->add_neighbor:4', 'decrease_a3->pdcch_2sym:3', 'insufficient_data->check_server:2', 'insufficient_data->pdcch_2sym:1']

### knn_conf60
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_knn_conf60.csv`
- changed rows vs base: 60
- F1/F1_5BS distribution: {'decrease_a3': 61, 'insufficient_data': 45, 'add_neighbor': 48, 'lift_tilt': 69, 'pdcch_2sym': 30, 'adjust_azimuth': 15, 'check_server': 1}
- top transitions: ['adjust_azimuth->add_neighbor:31', 'adjust_azimuth->decrease_a3:14', 'decrease_a3->add_neighbor:4', 'lift_tilt->insufficient_data:4', 'decrease_a3->pdcch_2sym:3', 'insufficient_data->lift_tilt:2', 'insufficient_data->check_server:1', 'insufficient_data->pdcch_2sym:1']

### knn_conf70
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_knn_conf70.csv`
- changed rows vs base: 22
- F1/F1_5BS distribution: {'decrease_a3': 58, 'insufficient_data': 45, 'adjust_azimuth': 43, 'lift_tilt': 71, 'add_neighbor': 25, 'pdcch_2sym': 27}
- top transitions: ['adjust_azimuth->decrease_a3:9', 'adjust_azimuth->add_neighbor:8', 'decrease_a3->add_neighbor:4', 'decrease_a3->pdcch_2sym:1']

### adj_only_knn50_else_candidate
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_adj_only_knn50_else_candidate.csv`
- changed rows vs base: 60
- F1/F1_5BS distribution: {'decrease_a3': 78, 'insufficient_data': 45, 'add_neighbor': 49, 'lift_tilt': 71, 'pdcch_2sym': 26}
- top transitions: ['adjust_azimuth->add_neighbor:36', 'adjust_azimuth->decrease_a3:24']

### adj_only_candidate_tau70
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_adj_only_candidate_tau70.csv`
- changed rows vs base: 6
- F1/F1_5BS distribution: {'decrease_a3': 60, 'insufficient_data': 45, 'adjust_azimuth': 54, 'lift_tilt': 71, 'pdcch_2sym': 26, 'add_neighbor': 13}
- top transitions: ['adjust_azimuth->decrease_a3:6']

### adj_only_plus_calibrated_knn60
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_adj_only_plus_calibrated_knn60.csv`
- changed rows vs base: 68
- F1/F1_5BS distribution: {'decrease_a3': 74, 'insufficient_data': 49, 'add_neighbor': 53, 'lift_tilt': 67, 'pdcch_2sym': 26}
- top transitions: ['adjust_azimuth->add_neighbor:36', 'adjust_azimuth->decrease_a3:24', 'decrease_a3->add_neighbor:4', 'lift_tilt->insufficient_data:4']

### hardblock_adjust_knn50_else_candidate
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_hardblock_adjust_knn50_else_candidate.csv`
- changed rows vs base: 107
- F1/F1_5BS distribution: {'decrease_a3': 67, 'insufficient_data': 64, 'add_neighbor': 57, 'pdcch_2sym': 30, 'lift_tilt': 47, 'check_server': 4}
- top transitions: ['adjust_azimuth->add_neighbor:39', 'lift_tilt->insufficient_data:28', 'adjust_azimuth->decrease_a3:21', 'insufficient_data->lift_tilt:6', 'decrease_a3->add_neighbor:4', 'decrease_a3->pdcch_2sym:3', 'insufficient_data->check_server:2', 'lift_tilt->check_server:1', 'lift_tilt->add_neighbor:1', 'decrease_a3->check_server:1', 'insufficient_data->pdcch_2sym:1']

### knn_conf50_plus_all_check_votes
- file: `results/phase2/f1_distribution_repair/submission_600_f1repair_knn_conf50_plus_all_check_votes.csv`
- changed rows vs base: 86
- F1/F1_5BS distribution: {'decrease_a3': 60, 'insufficient_data': 54, 'add_neighbor': 53, 'lift_tilt': 58, 'pdcch_2sym': 30, 'adjust_azimuth': 10, 'check_server': 4}
- top transitions: ['adjust_azimuth->add_neighbor:36', 'lift_tilt->insufficient_data:18', 'adjust_azimuth->decrease_a3:14', 'insufficient_data->lift_tilt:6', 'decrease_a3->add_neighbor:4', 'decrease_a3->pdcch_2sym:3', 'insufficient_data->check_server:2', 'lift_tilt->check_server:1', 'decrease_a3->check_server:1', 'insufficient_data->pdcch_2sym:1']

