import sdl_agents



test = sdl_agents.AutoGenSystem('ArgoLLMs',
                                        's26_workdir',
                                        's26_commands/S26_commandline_full.py',)
test.register_local_agents()


request= "Move the sample up by 100"
response = test.initiate_chat(request)
print(response)