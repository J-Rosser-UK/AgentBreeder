from base import Population, Framework
from chat import get_structured_json_response_from_gpt
import logging

class Illuminator():

    def __init__(self, args):

        self.args = args

    def illuminate(self, population:Population, frameworks_for_evaluation:Framework):


        generation = population.generations[-1] 

      
        illuminated_frameworks_for_evaluation = []

        
        print(f"Number of clusters in generation {generation.generation_id}: {len(generation.clusters)}")
       
        for framework in frameworks_for_evaluation:

            new_framework = {"framework_name": framework.framework_name, "framework_thought_process": framework.framework_thought_process, "framework_code": framework.framework_code}


            framework_cluster_id = framework.cluster_id

            framework_cluster = framework.cluster

            print(framework_cluster_id, framework_cluster)


            cluster_frameworks:list[Framework] = [
                {"framework_name": fw.framework_name, "framework_thought_process": fw.framework_thought_process, "framework_code": fw.framework_code, "framework_fitness": fw.framework_fitness}
                 
                 for fw in framework.cluster.frameworks if str(fw.framework_id) != str(framework.framework_id)]
            
            
            if len(cluster_frameworks) <= 1:
                illuminated_frameworks_for_evaluation.append(framework)
                continue


            response_format = {
                "thinking": "Your step by step thinking.",
                "Will it outperform the other frameworks?": "A single letter, Y or N."
            }
            
            messages = [
                {"role": "system", "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""},
                {"role": "user", "content": f"""
                    Given the following multi-agent frameworks defined in code, do you think there is a 50% or greater chance that the
                    new framework will outperform or equally perform against the best of them in a simulated environment? Please be generous and optimistic.
                
                    Here are the frameworks in the cluster:
                    {cluster_frameworks}

                    Here is the new framework:
                    {new_framework}

                    Please use the following response format:
                    {response_format}
                    
                    """.strip() 
                
                },
            ]
            
            
            try:
                Y_or_N = get_structured_json_response_from_gpt(messages, response_format, model=self.args.model, temperature=0.5)
                print(Y_or_N)
                if Y_or_N["Will it outperform the other frameworks?"] == "Y":
                    illuminated_frameworks_for_evaluation.append(framework)
            except Exception as e:
                logging.error(f"Error in illumination: {e}")
                continue

        return illuminated_frameworks_for_evaluation