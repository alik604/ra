# MCTS based follow ahead RL

## Methods

This is in [Google doc](https://docs.google.com/document/d/11x_Wpk4UQDjVFefjjUeyngc1KjLbhyIsAlf1_7Y_KVo/edit?usp=sharing) - Email me to be added as a editor. Only suggestions allowed, due to public facing link.

## Key files

### `MCTS.py`

> see in-line documentation and docstrings

#### Nuances & Misc

##### Bugs I think I have

I think I am using the generated trajectories worng. see `move_test.py`. I think `discrete_action_space.pickle` should be centered about (0,0), I think Payamn said otherwise, but I'm switching to my resoning in a last ditch hope.

It is going to be very important to test/visualize the trajectories, this is being done in `move_test.py` to see if they can be done without falling. **There is a bug here**, I have played around with when to and to-not make pose relative to the robot. The falling might be due to rounding, do ing the trajecty generations (pre-computing) phase. 

I disabled falling and trained overnight, which was a mistake .

##### Generating the trajectories

in navagation.launch add underscore to the followign line as such, `<remap from="/cmd_vel" to="/tb3_$(arg agent_num)/jackal_velocity_controller/cmd_vel_____" />`, for generating the trajectories

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
* Do a `ctrl+f` and search `TODO`, I noted many of the tempory chages, such as logging, reward related

### Miscellaneous

The file `rnn_single_threaded_ros.py`, is based on `rnn.py` which uses multiprocessing. This is world models (without the CNN-autoencoder), which was My and Emma's contribution to our CMPT 419 group project.

Anthony's work was on ROS and taking my HINN (Human's next `x, y, theta` prediction given state, with is based on a laser scan) and using it to follow from ahead.

#### How to run

> Please note you might have to clone this repo with `git submodule update --init --recursive`, so get the contents of `/multi_jackal`.

Lauching...

1. launch_tb
2. launch_nav
3. launch_scripts  -> `python ../old_scripts/tf_node.py`
4. launch_scripts  -> `python3 test_move.py`

Where `/home/alik604/ra` is the path to the project directory.

```bash
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash

alias launch_tb="cd ~/ra/far_ws && . devel/setup.bash && cd /home/alik604/ra/far_ws/ && roslaunch src/follow_ahead_rl/launch/turtlebot.launch"
alias launch_nav="cd ~/ra/far_ws && . devel/setup.bash && cd /home/alik604/ra/far_ws/ && roslaunch src/follow_ahead_rl/launch/navigation.launch"
alias launch_scripts="cd ~/ra/far_ws && . devel/setup.bash && cd /home/alik604/ra/far_ws/src/follow_ahead_rl/scripts && conda deactivate"
```

This is faster than the manual process...

```bash
catkin_make
. devel/setup.bash
roslaunch src/follow_ahead_rl/launch/turtlebot.launch
```
