# MCTS based follow ahead RL

## Key files

### `MCTS.py`

> see in-line documentation and docstrings

#### Nuances & Misc

##### Ephemeral state

`env.robot_simulated` and `env.person_simulated` have been added. Their state should be treated as ephemeral. in every simulation, the state is inherited from the real state, and is hence incorrect. This _state_ is then over written completely whenever `env.get_observation_relative_robot(states_to_simulate_robot, states_to_simulate_person)` is called, even if it is without pramaters.
    * velocity_history and orientation_history are not actually used.

##### Ordering

* There are two `deque` objects to have a rolling window. You append to the left, and the right-most element is ejected if the list is too long.  

### `precompute_trajectories.py`

* simple code to call `env.build_action_discrete_action_space()`, which saves a pickle file to disk with the name `discrete_action_space.pickle`.

### `hinn_data_collector.py` & `hinn_train.py`

[goal] To predict the next person `[x, y, orianatation]` given the past few `[x, y, orianatation]`.

* Will collect a 4 tupple, where each element is the list `[x, y, orianatation]`. We are appending to the left, so the 0th index is the latest.
* Then we will train a regression model on the above data.
  * Implemntated models are: MLP, Catboost, Random Forest Regressor, and the classic Polynomial regression

> the data is saved in a pickle file. There are two small code issues, to avoid failed saving, or double saving (one checkpoint, one rolling latest), the file must be manualy renamed

### `... /gym_gazeboros_ac.py`

> Some of the major changes are:

* `build_discrete_action_space`: imagine 3 nested circles, but they are discrete. thre are 3 radai, and N number of tickers. This function calls `set_goal` N+1 times. and once at start to _prime_, this bug is easily ignored.
* `set_goal`: takes `orientation, x, y, z`, and **Publishes** to `/move_base_simple/goal_0`. Next this function **Subscribes** to `/move_base_node_0/TebLocalPlannerROS/local_plan` and has a callback `path_cb`
* `path_cb`: this callback takes a message, this first time it is called, it will iterate through the poses, which are the points the robot should be directed to, and...
  * Convert Quaternion to Euler (orientation)
  * convert [x, y, orientation] to relative to robot.
    * By use of 3 queue ojects, the pose is propagated to `build_discrete_action_space`.
* `History()`: I had implemented a deep copy, but found out that the History class was not a mere Queue object, but rather selective/discriminative as it considered frame rate and timing (`save_rate`). In a sense this is downsampling data. add_element is the problum.
  * I ended up using a deque with `maxlen` (_windows size_) instead.  

### Miscellaneous

The file `rnn_single_threaded_ros.py`, is based on `rnn.py` which uses multiprocessing. This is world models (without the CNN-autoencoder), which was My and Emma's contribution to our CMPT 419 group project.

Anthony's work was on ROS and taking my HINN (Human's next `x, y, theta` prediction given state, with is based on a laser scan) and using it to follow from ahead.
