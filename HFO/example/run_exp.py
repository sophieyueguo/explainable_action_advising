import subprocess
import shlex
import time


# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 2001 2 2 NoAdvise'))
# print ('No Advise Finished')
# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 2001 2 2 Early'))
# print ('Early Finished')
# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 2001 2 2 Alternative'))
# print ('Alternative Finished')
# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 2001 2 2 Importance'))
# print ('Importance Finished')
#
# start = time.time()
# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 2001 2 2 MistakeCorrecting'))
# end = time.time()
# print('Time Spent', end - start)



# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 201 2 2'))




subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 2001 2 2 Early'))
# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 2001 2 2 Alternative'))
# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 3 2 2 Importance'))
# subprocess.call(shlex.split('./simulate_python_sarsa_agents.sh 3 2 2 MistakeCorrecting'))
