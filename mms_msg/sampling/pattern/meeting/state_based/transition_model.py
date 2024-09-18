from __future__ import annotations
import json
import numpy as np
from typing import Optional, List, Union, Generic, TypeVar, Tuple, Any, Set, Dict
from abc import ABC, abstractmethod
from copy import deepcopy
import sys

from mms_msg.sampling.pattern.meeting.scenario_sequence_sampler import sample_balanced


class StateTransitionModel(ABC):
    """
    Abstract class that can be used to implement models which determine a sequence of previously defined states.
    """

    @abstractmethod
    def next(self, rng: np.random.random = np.random.default_rng()) -> Any:
        """
        Returns the next state according to the implemented model.

        Args:
            rng: rng that can be used in the determination of the next state

        Returns: Next state
        """
        raise NotImplementedError

    @abstractmethod
    def step_back(self) -> bool:
        """
        Attempt to revert the last step of the model. Only one successful step back is guaranteed.

        Returns: True when successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the transition model to its starting state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_json(obj: StateTransitionModel) -> str:
        """ Static method that serializes a StateTransitionModel into a json string.

        Args:
            obj: StateTransitionModel which should be serialized

        Returns: Json string which contains the data of the given object
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_json(json_string: str) -> StateTransitionModel:
        """ Static method that creates a StateTransitionModel from a given json string.

        Args:
            json_string: Json string that contains the data required for the StateTransitionModel

        Returns: StateTransitionModel constructed from the data of the json string.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save(obj: StateTransitionModel, filepath: Optional[str] = 'state_transition_model.json') -> None:
        """ Static method that saves the given StateTransitionModel to a file belonging to the given filepath.
            When the file exists its contests will be overwritten. When it not exists it is created.
            The used dataformat is json, so a .json file extension is recommended.

            Args:
                obj: StateTransitionModel which should be saved
                filepath: Path to the file where the model should be saved.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(filepath: Optional[str] = 'distribution_model.json') -> StateTransitionModel:
        """Static method that loads a StateTransitionModel from file belonging to the given filepath.

        Args:
            filepath: Path to the file where the model is saved.

        Returns: StateTransitionModel constructed from the data of the given file.
        """
        raise NotImplementedError


MST = TypeVar('MST')  # MarkovStateType


class MarkovModel(StateTransitionModel, Generic[MST]):
    """
    StateTransitionModel which uses a markov model for the state transitions.

    Properties:
        s0: Name or index of the starting state
        size: Number of states in the model
        states: (read-only) Names of the states
        current_state_index: Index of the current state
        last_state_index: Index of the previous state
    """

    def __init__(self, probability_matrix: np.ndarray, s0: Optional[Union[int, MST]] = 0,
                 state_names: Optional[List[MST]] = None) -> None:
        """
        Initialize a Markov Model with n states. The states can be named with the state_names parameter.
        Default naming is a simple enumeration.

        Args:
            probability_matrix: transition matrix for the markov model.
                It must be a stochastic matrix of the dimensions n x n.
            s0: index of the starting state or name of the starting state.
                When s0 is an integer s0 it is interpreted as the index of the starting state.
            state_names: (optional) names of the states, if used all states have to be named.
                As name every type except integer can be used,
                this type is reserved for the default enumeration.
        """

        # Assert correct dimensions and correct probabilities
        if not probability_matrix.shape[0] > 0:
            raise ValueError('The markov model requires at least one state')
        if probability_matrix.shape[0] != probability_matrix.shape[1]:
            raise ValueError(f'Probability matrix has wrong dimensions: Expecting a square matrix, '
                             f'but got {probability_matrix.shape}.')
        if not np.allclose(np.sum(probability_matrix, axis=1), np.ones(probability_matrix.shape[0]),
                           rtol=0.0, atol=1e-06):
            raise ValueError('The probabilities in some row, are not adding up to 1')

        # Assert that when states are named, all states have names
        if not (state_names is None or len(state_names) == probability_matrix.shape[0]):
            raise ValueError('When the states parameter is used, there has to be names for all states.')

        # Assert that no integers are used as names
        if state_names is not None and type(state_names[0]) is int:
            raise TypeError('Integers are not allowed as names for states')

        self.s0 = s0
        self._size = probability_matrix.shape[0]
        self.probability_matrix = probability_matrix
        if state_names is None:
            self._state_names = [*range(self.size)]
        else:
            self._state_names = deepcopy(state_names)

        if isinstance(s0, int):
            self.current_state_index = s0
        else:
            self.current_state_index = self.state_names.index(s0)

        self.last_state_index = None

    def next(self, rng: np.random.random = np.random.default_rng()) -> Union[int, MST]:
        """
        Simulates a step in the markov model and returns the state after the step

        Args:
            rng: The numpy rng that should be used should generate a number in the interval [0,1).
                When not set a uniform rng is used.

        Returns: Name of the state of the model after the step is executed
        """

        self.last_state_index = self.current_state_index

        self.current_state_index = self._simulate_step(self.current_state_index, rng)

        return self.state_names[self.current_state_index]

    def step_back(self) -> bool:
        """
        Attempt to revert the last step of the model. Only the last previous state is saved and can be restored.

        Returns: True when successful, False otherwise
        """
        if self.last_state_index is not None:
            self.current_state_index = self.last_state_index
            self.last_state_index = None
            return True
        else:
            return False

    def reset(self) -> None:
        """
        Resets the current state of the model to the starting state.
        """
        if isinstance(self.s0, int):
            self.current_state_index = self.s0
        else:
            self.current_state_index = self.state_names.index(self.s0)

    def simulate_step(self, state: Union[int, MST], rng: np.random.random = np.random.default_rng()) -> Union[int, MST]:
        """
        Simulates a step from a single state, without changing to current state.

        Args:
            state: Index or name of the starting state for the single step
            rng: The numpy rng that should be used, the rng should generate a number in the interval [0,1).
                If not set a uniform rng is used.

        Returns: State after a single step from the starting state
        """

        index = self.state_names.index(state)
        return self.state_names[self._simulate_step(index, rng)]

    def _simulate_step(self, index: int, rng: np.random.random = np.random.default_rng()) -> int:
        """
        Internal function that simulates one step from the given starting state, without changing the current state.
        In the input and output of this function, only the index of the states is used to identify them.

        Args:
            index: Index of the starting state for the single step
            rng: The numpy rng that should be used should generate a number in the interval [0,1).
                 If not set a uniform rng is used.

        Returns: Index of the State after a single step from the starting state
        """

        row = self.probability_matrix[index]
        random = rng.random()

        row_sum = 0
        for i, v in enumerate(row):
            row_sum += v
            if row_sum >= random:
                return i

    @property
    def size(self) -> int:
        return self._size

    @property
    def state_names(self) -> List[Union[int, MST]]:
        return self._state_names

    def __repr__(self):
        return (
            "Markov model: "
            f"Number of states: {self._size} "
            f"State names: {self._state_names}\n"
            f"Probability matrix:\n {self.probability_matrix}"
        )

    @staticmethod
    def to_json(obj: MarkovModel) -> str:
        def json_default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            else:
                return o.__dict__
        return json.dumps(obj, default=json_default)

    @staticmethod
    def from_json(json_string: str) -> MarkovModel:
        obj = MarkovModel(np.ones((1, 1)))
        data = json.loads(json_string)
        for k, v in data.items():
            if k == 'probability_matrix':
                obj.__dict__[k] = np.asarray(v)
            else:
                obj.__dict__[k] = v
        return obj

    @staticmethod
    def save(obj: MarkovModel, filepath: Optional[str] = 'state_transition_model.json') -> None:
        with open(filepath, 'w+') as file:
            try:
                json_string = MarkovModel.to_json(obj)
                file.write(json_string)
            finally:
                file.close()

    @staticmethod
    def load(filepath: Optional[str] = 'state_transition_model.json') -> MarkovModel:
        with open(filepath, 'r') as file:
            try:
                json_string = file.read()
                obj = MarkovModel.from_json(json_string)
            finally:
                file.close()
            return obj


class SpeakerTransitionModel(ABC):
    """
    Abstract class that can be used to implement a model that can sample a sequence of active speakers,
    along with an action for each speaker. For example, the action can be used  to determine the type of transition
    between the speakers.
    """
    @abstractmethod
    def start(self, env_state: Optional[Any] = None, **kwargs) -> Tuple[int, Any]:
        """
        Returns the speaker that should start speaking.
        This method should be called when a new sequence is sampled.

        Args:
            env_state: env_state: (optional) Additional information about the state of the environment

        Returns: Tuple consisting of the starting state of the next sequence and environment state
        """
        raise NotImplementedError

    @abstractmethod
    def next(self, rng: np.random.random = np.random.default_rng(), last_action_success: bool = True,
             env_state: Optional[Any] = None, **kwargs) -> Tuple[str, int, Any]:
        """
        Returns the next action and the next speaker according to the implemented transition model.
        When the model cannot find an action that can be executed a StopIteration error is raised.

        Args:
            rng: rng that can be used in the determination of the next state
            last_action_success: (optional) status of the execution of the last action,
                                  can be used when the actions can fail.
            env_state: (optional) Additional information about the state of the environment

        Returns: Next action (state), the index of the next active speaker and the current environment state
        """
        raise NotImplementedError

    @abstractmethod
    def change_num_speakers(self, num_speakers: int = 2) -> None:
        """
        Tries to change the number of speakers in the transition model.
        This can be used to create meetings with a different number of speakers than in the source dataset.
        When the change is not possible due to the structure of the transition model,
        this function should throw an SystemError.

        Args:
            num_speakers: New number of speakers that the transition model should use for its output.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the transition model to its initial state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def tags(self) -> Set[str]:
        """
        Tag of the transition model, the tag consists of a set of all actions that can be returned.

        Returns: Set of all possible actions.
        """
        pass

    @staticmethod
    @abstractmethod
    def to_json(obj: SpeakerTransitionModel) -> str:
        """ Static method that serializes a SpeakerTransitionModel into a json string.

        Args:
            obj: SpeakerTransitionModel which should be serialized

        Returns: Json string which contains the data of the given object
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_json(json_string: str) -> SpeakerTransitionModel:
        """ Static method that creates a SpeakerTransitionModel from a given json string.

        Args:
            json_string: Json string that contains the data required for the SpeakerTransitionModel

        Returns: SpeakerTransitionModel constructed from the data of the json string.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save(obj: SpeakerTransitionModel, filepath: Optional[str] = 'speaker_transition_model.json') -> None:
        """ Static method that saves the given SpeakerTransitionModel to a file belonging to the given filepath.
            When the file exists its contests will be overwritten. When it not exists it is created.
            The used dataformat is json, so a .json file extension is recommended.

            Args:
                obj: SpeakerTransitionModel which should be saved
                filepath: Path to the file where the model should be saved.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(filepath: Optional[str] = 'speaker_transition_model.json') -> SpeakerTransitionModel:
        """Static method that loads a SpeakerTransitionModel from file belonging to the given filepath.

        Args:
            filepath: Path to the file where the model is saved.

        Returns: SpeakerTransitionModel constructed from the data of the given file.
        """
        raise NotImplementedError


class MultiSpeakerTransitionModel(SpeakerTransitionModel):
    """SpeakerTransitionModel which implements the transition between multiple speakers.
    For the transition of the states a transitionModel is used.
    Supports four possible actions:
        - TH: Turn-hold     (no speaker change with silence)
        - TS: Turn-switch   (speaker change with silence)
        - OV: Overlap       (speaker change with overlap)
        - BC: Backchannel   (speaker in backchannel, totally overlapped from foreground speaker)

    Currently, there are two implemented modes, that determine the selection of the next speaker (random, balanced)

    Corresponding paper: Improving the Naturalness of Simulated Conversations for End-to-End Neural Diarization,
                         https://arxiv.org/abs/2204.11232

    Properties:
        transition_model: StateTransitionModel used for the selection of the action
        current_active_index: Index of current active speaker
        last_active_index: Index of last active speaker
        tries: Current number of tries for finding an action that can be successfully executed
        max_tries: Maximum number of tries to find a valid action,
                   after this is surpassed a StopIteration Exception is returned
        tags: Set of all actions that can be returned by the model.
        num_speakers: Number of speakers in the transition model.
        mode: Currently, there are two implemented modes, that determine the selection of the next speaker:
            - random:   The next speaker is chosen at random.
            - balanced: The next speaker is the speaker which has the least amount of speech time in the
                        currently generated section of the meeting.

    """
    def __init__(self, transition_model: StateTransitionModel, max_tries: int = 50, num_speakers: int = 2,
                 mode: str = 'balanced') -> None:
        """
        Initialization with a StateTransitionModel and the number of maximal tries for a successful action.
        When it is not possible to execute the action, the StateTransitionModel is reverted
        and the selection of the next step is done again.
        When more than max_tries are required a StopIteration Exception is returned

        Args:
            transition_model: Underlying transition model, that is internally used to determine the next action.
            max_tries: Maximum number of tries
            num_speakers: Number of speakers in the transition model
        """

        self.transition_model = transition_model
        self.current_active_index = 0
        self.last_active_index = 0  # Required to revert internal state after sampling of a fitting source fails.
        self.tries = 0
        self.max_tries = max_tries
        self.num_speakers = num_speakers
        self.mode = mode

    def start(self, env_state: Optional[Any] = None, **kwargs) -> Tuple[int, Any]:
        self.reset()
        return self.current_active_index, env_state

    def next(self, rng: np.random.random = np.random.default_rng(), last_action_success: bool = True,
             env_state: Optional[Any] = None, examples: List[Dict] = None, **kwargs) -> Tuple[str, int, Any]:
        """
        Returns the next action and the next speaker according to the implemented transition model.
        When the model cannot find an action that can be executed a StopIteration error is raised.

        Args:
            rng: rng that can be used in the determination of the next state
            last_action_success: (optional) status of the execution of the last action,
                                  can be used when the actions can fail.
            env_state: (optional) Additional information about the state of the environment
            examples: (optional) List of the previously chosen samples of the current meeting

        Returns: Next action (state), the index of the next active speaker and the current environment state
        """

        if examples is None:
            examples = []

        if last_action_success:
            self.tries = 0
        elif self.tries < self.max_tries:
            self.transition_model.step_back()
            self.current_active_index = self.last_active_index
            self.tries += 1
        else:
            raise StopIteration('Number of max tries exceeded')

        action = self.transition_model.next(rng)

        self.last_active_index = self.current_active_index

        # In the case of a backchannel, the next speaker changes, but the active speaker in the foreground does not.
        if action == "BC":
            return action, self._next_speaker(rng=rng, examples=examples, env_state=env_state, **kwargs), None

        # Speaker changes in the case of a turn-switch or overlap
        if action in ("TS", "OV"):
            self.current_active_index = self._next_speaker(rng=rng, examples=examples, env_state=env_state, **kwargs)

        return action, self.current_active_index, env_state

    def change_num_speakers(self, num_speakers: int = 2):
        self.num_speakers = num_speakers

    def _next_speaker(self, rng: np.random.random, examples: List[Dict], **kwargs) -> int:
        """
        Internal function that determines the next speaker, depending on the current mode.

        Args:
            rng:  rng that can be used in the determination of the next speaker.
            examples: List of the previously chosen samples of the current meeting.
            **kwargs: Additional keyword arguments given to the function.

        Returns: Index of the next speaker.
        """
        if self.mode == 'balanced':
            speakers = set()

            for source in examples:
                speakers.add(source['speaker_id'])

            speakers = list(speakers)
            if len(speakers) < self.num_speakers:
                speakers.extend([str(i) for i in range(len(speakers), self.num_speakers)])

            next_index = speakers.index(str(sample_balanced(scenarios=speakers, examples=examples, rng=rng)))

            # When the active speaker is selected, a random speaker is chosen,
            # because the selected action requires a speaker change,
            if next_index != self.current_active_index:
                return next_index
            else:
                possible_speakers = list({i for i in range(self.num_speakers)}.difference({self.current_active_index}))
                return int(rng.choice(possible_speakers, size=1))

        elif self.mode == 'random':
            possible_speakers = list({i for i in range(self.num_speakers)}.difference({self.current_active_index}))
            return int(rng.choice(possible_speakers, size=1))
        else:
            raise AssertionError(f'Selected mode ({self.mode}) is not supported. Supported modes: random, balanced')

    def reset(self) -> None:
        self.transition_model.reset()
        self.current_active_index = 0
        self.last_active_index = 0

    @property
    def tags(self) -> Set[str]:
        return {"TS", "TH", "OV", "BC"}

    def __repr__(self):
        return (
            f"MultiSpeakerTransitionModel: "
            f"Current state: {self.current_active_index}"
            f"Number of speakers:{self.num_speakers}\n"
            f"TransitionModel:\n{self.transition_model}"
        )

    @staticmethod
    def to_json(obj: MultiSpeakerTransitionModel) -> str:
        def json_default(o):
            if isinstance(o, StateTransitionModel):
                return type(o).__name__, o.__class__.to_json(o)
            else:
                return o.__dict__

        return json.dumps(obj, default=json_default)

    @staticmethod
    def from_json(json_string: str) -> MultiSpeakerTransitionModel:
        obj = MultiSpeakerTransitionModel(MarkovModel(np.ones((1, 1))))
        data = json.loads(json_string)
        for k, v in data.items():
            if k == 'transition_model':
                # Restriction of possible classes to the ones in this file to prevent
                # possible injection of malicious code through the json string
                obj.__dict__[k] = getattr(sys.modules[__name__], v[0]).from_json(v[1])
            else:
                obj.__dict__[k] = v
        return obj

    @staticmethod
    def save(obj: MultiSpeakerTransitionModel, filepath: Optional[str] = 'speaker_transition_model.json') -> None:
        with open(filepath, 'w+') as file:
            try:
                json_string = MultiSpeakerTransitionModel.to_json(obj)
                file.write(json_string)
            finally:
                file.close()

    @staticmethod
    def load(filepath: Optional[str] = 'speaker_transition_model.json') -> MultiSpeakerTransitionModel:
        with open(filepath, 'r') as file:
            try:
                json_string = file.read()
                obj = MultiSpeakerTransitionModel.from_json(json_string)
            finally:
                file.close()
            return obj
