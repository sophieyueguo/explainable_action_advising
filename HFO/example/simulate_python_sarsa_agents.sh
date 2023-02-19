#! /bin/sh

#This script calls the python implementation of the high_level_sarsa_agent
#In essence it calls the relevant functions from a thin python wrapper written over the C++ sarsa_libraries

# HOW TO RUN
#takes in the number of trails as first argument
#takes in the number of offense agents as second argument
#takes in the number of defense agents as the third argument
# eg. if one needs to run 200 episodes of 2v2 then execute
# ./simulate_python_sarsa_agents.sh 200 2 2

port=6000
trials=10000
oa=2  #number of offense agents
da=1  #number of defense agents
if [ "$#" -lt 1 ]
then
  :
else
  trials=$1
  oa=$2
  da=$3
  advice_strategy=$4
fi

#kill any other simulations that may be running
killall -9 rcssserver
sleep 2

cd ..         #cd to HFO directory
# rm weights*   #remove weights from old runs

# python="/usr/bin/python3.8" #which python? sophie: 3.8
python="python3.8" #which python? sophie: 3.8
agent_path="./example/sarsa_offense"
log_dir="."
output_path=$agent_path
agent_filename="high_level_sarsa_agent.py"

viper_path="./example/sarsa_offense/policy_extraction"
viper_filename="viper.py"

#start the server
# stdbuf -oL ./bin/HFO --port=$port --no-logging --offense-agents=$oa --defense-npcs=$da --trials=$trials --defense-team=base --fullstate  --headless > $log_dir/"$oa"v"$da""_sarsa_py_agents.log" &
stdbuf -oL ./bin/HFO --port=$port --no-logging --offense-agents=$oa --defense-npcs=$da --trials=$trials --defense-team=base --fullstate > $log_dir/"$oa"v"$da""_sarsa_py_agents.log" &

#each agent is a seperate process

for n in $(seq 1 $oa)
do
    sleep 5
    fname=$advice_strategy
    fname+="_agent"
    fname+=$n
    fname+=".txt"
    logfile=$log_dir/$fname
    rm $logfile
    # original script, does not take --advice_strategy
    # $python $agent_path/$agent_filename --port=$port --numTeammates=`expr $oa - 1` --numOpponents=$da --numEpisodes=$trials --suffix=$n &> $log_dir/$fname &

    # AA/EAA script
    # change the parameter of "use_EAA"
    $python $agent_path/$agent_filename --port=$port --numTeammates=`expr $oa - 1` --numOpponents=$da --numEpisodes=$trials --suffix=$n --advice_strategy=$advice_strategy  --use_EAA=True &> $log_dir/$fname &

    # viper script
    # $python $viper_path/$viper_filename --port=$port --numTeammates=`expr $oa - 1` --numOpponents=$da --numEpisodes=$trials --suffix=$n  &> $log_dir/$fname &
done

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
