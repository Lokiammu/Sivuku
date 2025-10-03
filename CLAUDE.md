# My preferences
My coding style is to NEVER use inline comments. Code should be well written and self explanatory. If it is not, then it should be refactored. I instead use private functions to explain the code, and use docstrings in the private functions to document what it is achieving if it's not obvious from the private function's name and signature.

In general, I am all for the KISS approach. I write opinionated code where there should be only ONE way of doing things, and it should be very minimal without any surprises. This will very much apply to this project.

# Context
What this project wants to achieve is to build a standard for agentic execution environments. We are partnering with HuggingFace on this, so we want to land in the Huggingface/PyTorch way of doing things (which is again, is minimal and very Pythonic).

What is an agentic execution environment? We have an opinionated answer for this: we start from the Gym/Gymnasium basic API of:

1. reset(): Start a new trajectory
2. step(): Takes an action, returns an observation (which could contain rewards)
3. render(): Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text.
4. close(): Closes the environment, important when external software is used, i.e. pygame for rendering, databases

As a reminder, we also follow the "classical RL" approach of having the environment hold the state, NOT the policy. So for example, if your policy is playing chess, the policy will give you a knee-jerk reaction to a board state and tell you the next move, but who's hosting and updating the board state is the environment. It will also be responsible for showing only the observable part of the state in e.g. partial information games like poker, or in a driver sim if a car is hidden behind buildings/other cars etc. Not every env will have a state, but if a state exists it will live there.

We are however going to make a few changes on top of it. We follow in the footsteps of TorchRL and bring over the following ideas:

1. We connect it to dataloaders. TorchRL uses the [DataCollector abstraction](https://docs.pytorch.org/rl/main/reference/collectors.html) to loop a policy and its environment together to build the "inner loop" where the env gets actions and the policy gets observations *and* (crucially) to take a dataloader to load the tasks. This is a genius idea because it allows us to encapsulate all the complexity of dynamic data generation in online RL and see "just a fancy dataloader". The negative is that this only works for the narrow application of RL post-training of LLMs, doesn't yet go to agents. So we are gonna have to extend it.

2. We compute rewards by attaching a `transform()` function to the Env's `step()`. This is a very flexible way of doing it, following in TorchVision's steps.

Finally, we extend to fully-fledged agents. We follow the same approach as CodeAct. Read the paper in the PDF in this directory to make sure you understand it and !!CRITICAL!! ASK QUESTIONS IF YOU DO NOT. In short, every *action* is arbitrary Python code that gets executed and that can chain multiple tool calls in a single action (as opposed to the JSON-style traditional function calling way which would spend the whole action to do a single tool call).

How does this connect to our env? We run `step` and eval python code that gets sent us, see this picture:
![agentic_loop.png](./agentic_loop.png)

We do not do this here -- this is just the big picture for you to familiarize. We assume that our customers will do this and only this.

# What I'm giving you
I'm giving you a half-baked implementation which connects Gym and TorchRL that I built for RL post-training a while back. It's not even fully working, but it will give you an idea about what I'm doing and how I write code, what my preferences are etc.

# What I expect from you
I want you to take my half-baked implementation and graduate it to a CodeAct implementation that can do agents while still being able to support RL training as well (via transforms).

This should be a dynamic session: please ask questions whenever you are not sure about things and use me as your design partner in a pair design session.
