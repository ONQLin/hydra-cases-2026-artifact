"""Small NumPy oracles for KDA algebra; no quantization or hardware timing.

The recurrent oracle and chunk graph interpreter are deliberately independent.
They validate tensor values and state transitions, not the W8/A8 cost surrogate.
"""

import numpy as np


class RecurrentKDAReference:
    @staticmethod
    def run(q, k, v, gates, beta, state=None):
        heads, tokens, key_dim = q.shape
        state = np.zeros((heads, key_dim, v.shape[-1]), dtype=q.dtype) if state is None else state.copy()
        output = np.empty_like(v)
        for token in range(tokens):
            state *= np.exp(gates[:, token, :, None])
            prediction = np.einsum('hk,hkv->hv', k[:, token], state)
            update = beta[:, token, None]*(v[:, token]-prediction)
            state += k[:, token, :, None]*update[:, None, :]
            output[:, token] = np.einsum('hk,hkv->hv', q[:, token]/np.sqrt(key_dim), state)
        return output, state


class ChunkKDAInterpreter:
    """Execute the production chunk graph's nodes with reference array operations."""
    def __init__(self, program):
        self.program = program

    def run(self, q, k, v, gates, beta, state=None):
        h, t, key_dim = q.shape
        if state is None:
            state = np.zeros((h,key_dim,v.shape[-1]), dtype=q.dtype)
        self.values = dict(q=q, k=k, v=v, g=gates, beta=beta, state_before=state.copy())
        for node in self.program.graph.ordered_nodes():
            result = getattr(self, node.name)(*[self.values[key] for key in node.inputs])
            self.values[node.outputs[0]] = result
            expected = self.program.graph.tensors[node.outputs[0]].shape
            if result.shape != expected:
                raise AssertionError((node.name, result.shape, expected))
        return self.values['output'], self.values['state_after']

    def gate_prefix(self, gates):
        return np.cumsum(gates, axis=1)

    def key_products(self, k, gates, beta):
        h, t, _ = k.shape
        lower = np.zeros((h,t,t), dtype=k.dtype)
        for i in range(t):
            for j in range(i):
                lower[:,i,j] = beta[:,i]*np.sum(k[:,i]*k[:,j]*np.exp(gates[:,i]-gates[:,j]),axis=-1)
        return lower

    def scaled_rhs(self, k, v, gates, beta):
        return np.concatenate((k*np.exp(gates),v),axis=-1)*beta[...,None]

    def triangular_solve(self, lower, rhs):
        return np.linalg.solve(np.eye(lower.shape[-1],dtype=lower.dtype)[None]+lower, rhs)

    def state_residual(self, wu, state):
        return wu[...,self.program.key_dim:]-wu[...,:self.program.key_dim]@state

    def query_products(self, q, k, gates):
        h, t, d = q.shape
        scores = np.zeros((h,t,t),dtype=q.dtype)
        for i in range(t):
            for j in range(i+1):
                scores[:,i,j] = np.sum(q[:,i]*k[:,j]*np.exp(gates[:,i]-gates[:,j]),axis=-1)/np.sqrt(d)
        return scores

    def prior_output(self, q, gates, state):
        return (q*np.exp(gates)/np.sqrt(q.shape[-1]))@state

    def output(self, scores, residual, prior):
        return scores@residual+prior

    def state_update(self, k, gates, residual, state):
        carry = np.exp(gates[:,-1,:,None])*state
        keys = k*np.exp(gates[:,-1:]-gates)
        return carry+np.swapaxes(keys,-1,-2)@residual
