import json
import random
import os

# Get the absolute path of the current directory (where the script is running)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Path to the 'agents_output.json' file
file_path = os.path.join(current_dir, 'agents_output.json')

# Load the JSON data
with open(file_path, 'r') as file:
    agents_data = json.load(file)

def get_agent_data(agent_name):
    """Returns voice samples and one random system prompt for the given agent name."""
    if agent_name in agents_data:
        agent = agents_data[agent_name]
        
        # Get all voice samples for the agent
        voice_samples = agent.get("voice_samples", [])
        
        # Convert relative paths in voice_samples to absolute paths
        absolute_voice_samples = [
            os.path.join(current_dir, sample) if sample.startswith("./") else sample
            for sample in voice_samples
        ]
        
        # Get a random system prompt for the agent
        system_prompts = agent.get("system_prompts", [])
        
        if system_prompts:
            random_system_prompt = random.choice(system_prompts)
        else:
            random_system_prompt = None
        
        return absolute_voice_samples, random_system_prompt, agent.get("cartesia_voice_id")
    else:
        return None, None

# Example Usage (you can call this function from another script)
if __name__ == "__main__":
    agent_name = "Donald Trump"  # Replace with any agent name
    voice_samples, random_system_prompt = get_agent_data(agent_name)
    print(f"Voice Samples for {agent_name}: {voice_samples}")
    print(f"Random System Prompt for {agent_name}: {random_system_prompt}")
