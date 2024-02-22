import numpy as np
from typing import Optional, List, Union, Generic, TypeVar, Tuple, Any, Set
from abc import ABC, abstractmethod
from copy import deepcopy


class StateTransitionModel(ABC):
    """
    Abstract class that can be used to implement models, which determine a sequence of previous defined states.
    """

    @abstractmethod
    def next(self, rng: np.random.random = np.random.default_rng()):
        """
        Returns the next state according to the implemented model.

        :param rng: rng that can be used in the determination of the next state
        :return: Name of the next state
        """
        raise NotImplementedError

    @abstractmethod
    def step_back(self):
        """
        Attempt to revert the last step of the model. Only one successful step back is guarantied.
        :return: True when successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Resets the transition model to its starting state.
        :return:
        """
        raise NotImplementedError


MST = TypeVar('MST')  # MarkovStateType


class MarkovModel(StateTransitionModel, Generic[MST]):
    """
    StateTransitionModel which uses a markov model for the state transitions.

    Properties:
        s0: Name or index of the starting state
        size: Number of states in the model
        states: (read only) Names of the states
        current_state_index: Index of the current state
        last_state_index: Index of the previous state
    """

    def __init__(self, size: int, probability_matrix: np.ndarray, s0: Optional[Union[int, str]] = 0,
                 states: Optional[List[MST]] = None) -> None:
        """ Initialize a Markov Model with n states. The states can be named with the states parameter.
            Default naming is a simple enumeration.

        :param size: number of states in the markov model
        :param probability_matrix: transition matrix for the markov model.
               Must be a stochastic matrix of the dimensions n x n.
        :param s0: index of the starting state or name of the starting state.
                   When the s0 is an integer s0 it is interpreted as the index of the starting state.
        :param states: (optional) names of the states, if used all states has to be named.
                        As name every type except integer can be used,
                        this type is reserved for the default enumeration.
        """

        # Assert correct dimensions and correct probabilities
        if not size > 0:
            raise AssertionError('The markov model requires at least one state')
        if not probability_matrix.shape == (size, size):
            raise AssertionError('Probability matrix has wrong dimensions, expecting: ('
                                 + str(size) + ',' + str(size) + ')')
        if not np.allclose(np.sum(probability_matrix, axis=1), np.ones(size), rtol=0.0, atol=1e-06):
            raise AssertionError('The probabilities in some row, are not adding up to 1')

        # Assert that when states are named, all states have names
        if not (states is None or len(states) == size):
            raise AssertionError('When the states parameter is used, there has to be names for all states.')

        # Assert that no integers are used as names
        if states is not None and type(states[0]) is int:
            raise AssertionError('Integers are not allowed as names for states')

        self.s0 = s0
        self._size = size
        self.probability_matrix = probability_matrix
        if states is None:
            self._states = [*range(size)]
        else:
            self._states = deepcopy(states)

        if isinstance(s0, int):
            self.current_state_index = s0
        else:
            self.current_state_index = self.states.index(s0)

        self.last_state_index = None

    def next(self, rng: np.random.random = np.random.default_rng()) -> Union[int, MST]:
        """ Simulates a step in the markov model and return the state after the step

        :param rng: The numpy rng that should be used, the rng should generate a number in the interval [0,1).
                    If not set a uniform rng is used.
        :return: Name of the state of the model after the step is executed
        """
        self.last_state_index = self.current_state_index
        row = self.probability_matrix[self.current_state_index:self.current_state_index + 1]

        random = rng.random()

        row_sum = 0
        for i in range(len(row[0])):
            row_sum += row[0][i]
            if row_sum >= random:
                self.current_state_index = i
                return self.states[self.current_state_index]

    def step_back(self) -> bool:
        """ Attempt to revert the last step of the model. Only the last previous state is saved and can be restored.
        :return: True when successful, False otherwise
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
            self.current_state_index = self.states.index(self.s0)

    def simulate_step(self, state: Union[int, MST], rng: np.random.random = np.random.default_rng()) -> Union[int, MST]:
        """
        Simulates a step from a single state, without changing to current state.
        :param state: starting state for the single step
        :param rng: The numpy rng that should be used, the rng should generate a number in the interval [0,1).
                    If not set a uniform rng is used.
        :return: state after a single step from the starting state
        """
        index = self.states.index(state)
        row = self.probability_matrix[index:index + 1]

        random = rng.random()

        row_sum = 0
        for i in range(len(row[0])):
            row_sum += row[0][i]
            if row_sum >= random:
                return self.states[i]

    @property
    def size(self) -> int:
        return self._size

    @property
    def states(self) -> List[Union[int, MST]]:
        return self._states

    def __repr__(self):
        np.set_printoptions(suppress=True)
        ret = "Markov model: "
        ret += " Number of states: " + str(self._size)
        ret += " State names: " + str(self._states)
        ret += "\nProbability matrix:\n" + str(self.probability_matrix)
        np.set_printoptions(suppress=False)
        return ret


class SpeakerTransitionModel(ABC):
    """Abstract class that can be used to implements a model that can sample a sequence of active speaker,
       along with an action for each speaker. For example the action can be used  to determine the type of transition
       between the speakers."""
    @abstractmethod
    def start(self, env_state: Optional[Any] = None, **kwargs) -> Tuple[int, Any]:
        """
        Returns the speaker that should start speaking.
        This method should be called, when a new sequence is sampled.

        :param env_state: (optional) Additional information about the state of the environment
        :return starting state of the next sequence and environment state
        """
        raise NotImplementedError

    @abstractmethod
    def next(self, rng: np.random.random = np.random.default_rng(), last_action_success: bool = True,
             env_state: Optional[Any] = None, **kwargs) -> Tuple[str, int, Any]:
        """
        Returns the next action and the next speaker according to the implemented transition model.
        When the model cannot find an action that can be executed a StopIteration error is raised.

        :param rng: rng that can be used in the determination of the next state
        :param last_action_success: (optional) status of the execution of the last action,
                                     can be used when the actions can fail.
        :param env_state: (optional) Additional information about the state of the environment
        :return: Next action (state), the index of the next active speaker and the current environment state
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the transition model to its initial state.
        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def tags(self) -> Set[str]:
        """
        Returns the tag of the transition model, the tag consists of a  et of all actions that can be returned.
        :return: Set of all possible actions.
        """
        pass


class TwoSpeakerTransitionModel(SpeakerTransitionModel):
    """SpeakerTransitionModel which implements the transition between two speakers.
     For the transition of the states a transitionModel is used.
     Supports four possible actions:
        - TH: Turn-hold     (no speaker change with silence)
        - TS: Turn-switch   (speaker change with silence)
        - OV: Overlap       (speaker change with overlap)
        - BC: Backchannel   (speaker in backchannel, totally overlapped from foreground speaker)

    Corresponding paper: Improving the Naturalness of Simulated Conversations for End-to-End Neural Diarization,
                         https://arxiv.org/abs/2204.11232

    Properties:
        transition_model: StateTransitionModel used for the selection of the action
        current_active_index: Index of current active speaker
        tries: Current number of tries for finding an action that can be successful executed
        max_tries: Maximum number of tries to find a valid action,
                   after this is surpassed a StopIteration Exception is returned
        tags: Set of all actions that can be returned by the model.
    """
    def __init__(self, transition_model: StateTransitionModel, max_tries: int = 50) -> None:
        """ Initialization with a StateTransitionModel and the number of maximal tries for a successful action.
            When it is not possible to execute the action, the StateTransitionModel is reverted
            and the selection of the next step is done again.
            When more than max_tries are required a StopIteration Exception is returned
        :param transition_model: Underlying transition model, that is internally used to determine the next action.
        :param max_tries: Maximum number of tries
        """
        self.transition_model = transition_model
        self.current_active_index = 0
        self.tries = 0
        self.max_tries = max_tries

    def start(self, env_state: Optional[Any] = None, **kwargs):
        self.reset()
        return self.current_active_index, env_state

    def next(self, rng: np.random.random = np.random.default_rng(), last_action_success: bool = True,
             env_state: Optional[Any] = None, **kwargs) -> Tuple[str, int, Any]:
        if last_action_success:
            self.tries = 0
        elif self.tries < self.max_tries:
            self.transition_model.step_back()
            self.tries += 1
        else:
            raise StopIteration('Number of max tries exceeded')

        action = self.transition_model.next(rng)

        # In the case of a backchannel, the next speaker changes, but the active speaker in the foreground does not.
        if action == "BC":
            return action, (self.current_active_index + 1) % 2, None

        # Speaker changes in the case of a turn-switch or overlap
        if action in ("TS", "OV"):
            self.current_active_index = (self.current_active_index + 1) % 2

        return action, self.current_active_index, env_state

    def reset(self) -> None:
        self.transition_model.reset()
        self.current_active_index = 0

    @property
    def tags(self) -> Set[str]:
        return {"TS", "TH", "OV", "BC"}

    def __repr__(self):
        ret = "TwoSpeakerTransitionModel: "
        ret += " Current state: " + str(self.current_active_index)
        ret += "\nTransitionModel: " + str(self.transition_model)
        return ret
