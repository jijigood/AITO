# Phase 3 Smoke should_override Gate Eval

Evaluated: 200
JSON valid: 200/200 = 1.0000
Mapped OK: 200/200
Accepted by gate: 41/200
Label should_override: 46/200
Pred should_override: 41/200
should_override precision: 0.7805
should_override recall: 0.6957
C-code leak: 0/200
API errors: 0/200
Deterministic IOU: 0.7700
Mapped LoRA IOU: 0.9200
Gated arbiter IOU: 0.9200

## Per GT action
- add_neighbor: n=38, det=0.6842, mapped=0.6316, gated=0.6316, action_acc=0.6842, override_acc=0.5000
- check_server: n=38, det=1.0000, mapped=1.0000, gated=1.0000, action_acc=1.0000, override_acc=0.9474
- decrease_a3: n=35, det=0.9429, mapped=0.9429, gated=0.9429, action_acc=0.9429, override_acc=0.9429
- insufficient_data: n=32, det=0.0000, mapped=1.0000, gated=1.0000, action_acc=1.0000, override_acc=1.0000
- lift_tilt: n=26, det=1.0000, mapped=1.0000, gated=1.0000, action_acc=1.0000, override_acc=1.0000
- pdcch_2sym: n=31, det=1.0000, mapped=1.0000, gated=1.0000, action_acc=1.0000, override_acc=1.0000
