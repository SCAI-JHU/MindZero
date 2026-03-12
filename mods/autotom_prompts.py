import random
from collections import Counter
from functools import partial
from typing import Literal

from pydantic import BaseModel, Field


OBJECT_NAMES = (
    "apple",
    "chips",
    "condimentbottle",
    "cupcake",
    "cutleryfork",
    "plate",
    "pudding",
    "remotecontrol",
    "salmon",
    "waterglass",
    "wineglass",
)
TARGET_NAMES = ("coffeetable", "dishwasher", "fridge", "kitchentable", "stove")
TASK_NAMES = (
    "prepare_food",
    "put_dishwasher",
    "put_fridge",
    "setup_table",
    "watch_tv",
)  # !!! BE CONSISTENT WITH THE TRAINING PROBLEM !!!


class Object(BaseModel):
    type: Literal[OBJECT_NAMES]
    count: int = Field(..., ge=1)

    @staticmethod
    def to_counter(objects):
        counter = Counter()
        for obj in objects:
            counter[obj.type] += obj.count
        return counter

    @staticmethod
    def from_counter(counter):
        objects = []
        for obj_type, count in counter.items():
            if count > 0:
                objects.append(Object(type=obj_type, count=count))
        return objects


class Target(BaseModel):
    type: Literal[TARGET_NAMES]
    # preposition: str = Field(..., choices=["on", "inside"]) # * reduce planning space


class GoalParticle(BaseModel):
    task_name: Literal[TASK_NAMES]
    objects: list[Object] = Field(..., min_items=1)
    target: Target
    p: float = Field(..., ge=0, le=1, description="Probability of the goal proposal")

    def minus_objects(self, counter):
        self.objects = Object.from_counter(Object.to_counter(self.objects) - counter)

    def plus_objects(self, counter):
        self.objects = Object.from_counter(Object.to_counter(self.objects) + counter)

    def to_natlang(self):
        counter = Object.to_counter(self.objects)
        objects = [f"{cnt} {name}" for name, cnt in counter.items()]
        return f"({self.task_name}) put {', '.join(objects)} to {self.target.type}"


class GoalParticles(BaseModel):
    particles: list[GoalParticle]

    def normalize(self):
        partition = sum(particle.p for particle in self.particles)
        if partition == 0:
            return
        for particle in self.particles:
            particle.p /= partition

    def reweight(self, probs, normalize=True):
        for particle, p in zip(self.particles, probs):
            particle.p *= p
        if normalize:
            self.normalize()

    def merge_duplicates(self):
        unique_particles = {}
        for particle in self.particles:
            key = particle.to_natlang()
            if key in unique_particles:
                unique_particles[key].p += particle.p
            else:
                unique_particles[key] = particle
        self.particles = list(unique_particles.values())

    def filter_low_conf(self, thres, min_num, normalize=True):
        self.particles = sorted(self.particles, key=lambda x: x.p, reverse=True)
        self.particles = self.particles[:min_num] + list(filter(lambda x: x.p >= thres, self.particles[min_num:]))
        if normalize:
            self.normalize()

    def fill_particles(self, particles, max_particles, normalize=True):
        for particle in particles.particles:
            if particle.to_natlang() not in self.to_natlang().keys():
                self.particles.append(particle)

                if len(self.particles) == max_particles:
                    if normalize:
                        self.normalize()
                    break

    def to_natlang(self):
        contents = dict()
        for particle in self.particles:
            contents[particle.to_natlang()] = round(100 * particle.p, 1)
        return contents

    def minus_objects(self, counter):
        for particle in self.particles:
            particle.minus_objects(counter)

    def plus_objects(self, counter):
        for particle in self.particles:
            particle.plus_objects(counter)

    def probs_grab(self, in_log=False):
        probs = Counter()
        for particle in self.particles:
            for object in particle.objects:
                probs[object.type] += particle.p

        if in_log:
            for obj_type, prob in probs.items():
                probs[obj_type] = round(100 * prob, 1)

        return dict(probs.most_common())

    def probs_put(self, in_log=False):
        probs = Counter()
        for particle in self.particles:
            probs[particle.target.type] += particle.p

        if in_log:
            for obj_type, prob in probs.items():
                probs[obj_type] = round(100 * prob, 1)

        return dict(probs.most_common())

    def best_in_probs(self, probs):
        candidates = [x for x, p in probs.items() if p == max(probs.values())]
        if len(candidates) == 0:
            return None, 0
        else:
            answer = random.Random(0).choice(candidates)
            return answer, probs[answer]

    def best_grab(self):
        return self.best_in_probs(self.probs_grab(in_log=False))

    def best_put(self):
        return self.best_in_probs(self.probs_put(in_log=False))

    def __len__(self):
        return len(self.particles)


class Likelihood(BaseModel):
    likelihood: float = Field(..., ge=0, le=1, description="likelihood in float number between 0 and 1.")


# * p(goal | next_human_action, curr_env_state, curr_human_state, key_action_history)
propose = """\
Human has been working on a task of moving some objects to a target location. The task type can only be one of the following: setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

Your are a helpful assistant. In order to help human, please propose multiple hypotheses of [human's overall goal] (including both finished and potential future subgoals), base on the following information:

[current state]
{curr_env_state}

{curr_human_state}

[key action history]
{key_action_history}

[human's next action]
{next_human_action}

Hints:
- The task type is constant and the target location is unique, i.e., human will be consistently doing the same task (setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV) and put all objects to the same location.
- Please propose diverse goals in both object type and count.

Output Requirements:
Please provide a probability distribution over n={n} hypotheses of [human's overall goal] (including both finished and potential future subgoals).
Your response should include the probability distribution formatted according to this JSON schema: {schema}
"""
propose = partial(propose.format, schema=GoalParticles.model_json_schema())

# * p(curr_human_action | goal, curr_env_state, curr_human_state, key_action_history)
forward_likelihood = """\
Human has been working on a task of moving some objects to a target location.

Given the following information:

[current state]
{curr_env_state}

{curr_human_state}

[human's unfinished goals]
{goal}

Hints:
- If human holds nothing, human must grab a goal object, or walk towards a goal object or its room. **Please note: it is perfectly fine to grab any goal object, not necessarily the nearest one.**
- If human holds something, human must put the object to the target location, or walk towards the target location or its room.
- The action is unlikely if it contradicts the above rules.

Determine if the following statement is likely, and respond with only either A or B.
[human's next action] {next_human_action}
A) Likely.
B) Unlikely.
""".format

forward_likelihood_all_time = """
Human has been working on a task of moving some objects to a target location.

Given the following information:

[initial state]
{init_env_state}

{init_human_state}

[human's overall goal]
{goal}

Reasonable human actions follow the following rules:
- Human will walk towards goal objects, grab goal objects, then walk towards the target location and put goal objects to the target location, until all goal objects are put to the target location. Human can execute this process in any order. It implies that human will not necessarily grab the nearest goal object first.
- Human will grab only the remaining goal objects, and put something only to the target location. Human will not grab additional goal objects beyond the remaining goal objects.
- Human will walk towards only goal objects or the target location.
- Human can grab at most two goal objects at the same time.

Determine if the following human's action sequence is likely. It's unlikely if it contradicts the above rules.

Note: The human's action sequence could be either complete or partial.

[human's action sequence]
{key_action_history}

Respond with only a single capital letter A or B, representing A=Likely or B=Unlikely.
""".strip().format


prior_v1 = """
Human has been working on a task of moving some objects to a target location. The task type can only be one of the following: setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

Determine if the following task description "(<task_type>) put <objects> to <target_location>" is consistent. In other words, if "put <objects> to <target_location>" is semantically consistent with the task type.

For example, "put a salmon to the dishwasher" is inconsistent with "put_fridge" task type; "put a remotecontrol to the table" is inconsistent with "setup_table" task type.

The task description is: {goal}

Respond with only a single capital letter A or B, representing A=Consistent or B=Inconsistent.
""".strip().format


prior_v2 = """
Human has been working on a task of moving some objects to a target location. The task type can only be one of the following: setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

Determine if the following task is reasonable or not:
{goal}

Positive examples:
1. (setup_table) put 2 cutleryfork, 2 plate, 2 waterglass to coffeetable
   Rationale: setting up a dinner table for 2 people, where each person has 1 set of cutlery, plate, and waterglass.
2. (prepare_food) put 1 apple, 1 pudding to stove
   Rationale: preparing food on the stove is reasonable.

Negative examples:
1. (setup_table) put 2 cutleryfork, 8 plate to kitchentable
   Rationale - unreasonable object in number: the numbers of plate and cutleryfork are highly discrepant.
2. (put_dishwasher) put 1 salmon, 1 cupcake to dishwasher
   Rationale - unreasonable object in type: putting food to the dishwasher is not reasonable.
3. (put_fridge) put 2 apple to fridge
   Rationale - unreasonable object in diversity: in order to effectively help human, number of object types should be ranging from 2 to 4.
4. (prepare_food) put 1 cupcake to fridge
   Rationale - unreasonable target location: putting food to the fridge is not consistent with the task of preparing food.

Respond with only a single capital letter A or B, representing A=Reasonable or B=Unreasonable.
""".strip().format

prior_v3 = """
Human has been working on a task of moving some objects to a target location. The task type can only be one of the following: setting up a table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.

Determine if the following task is reasonable or not: {goal}

Check if the goal violates any of the following rules. Only check the following rules, do not check other.
1. **Not diverse in object type**
   E.g.: (put_fridge) put 2 apple to fridge
   Rationale: in order to effectively help human, number of object types should be ranging from 2 to 4.
2. **Discrepant in object number**
   E.g.: (setup_table) put 2 cutleryfork, 8 plate to kitchentable
   the numbers of plate and cutleryfork are highly discrepant.

Respond with only a single capital letter A or B, representing A=Compliant or B=Violation.
""".strip().format

# TODO: add prior_v4: forbid to propose objects not in the apartment

prior_by_version = {1: prior_v1, 2: prior_v2, 3: prior_v3}
