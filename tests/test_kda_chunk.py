"""KDA chunk algebra, partial chunks, tile residency and ordered spill checks."""

from dataclasses import replace
import unittest

import numpy as np

from Sim.config.attention_operator_config import KDAConfig
from analytic_profile.attention import KDAProfile
from analytic_profile.kda import BaseKDAProgram, ChunkKDAProgram, KDACoreSchedule
from analytic_profile.modern import AcceleratorRates
from tests.reference_kda import ChunkKDAInterpreter, RecurrentKDAReference


class KDAChunkTest(unittest.TestCase):
    def inputs(self, tokens, dtype=np.float64):
        rng = np.random.default_rng(19+tokens)
        q, k, v = [rng.normal(size=(3,tokens,5)).astype(dtype) for _ in range(3)]
        q /= np.linalg.norm(q,axis=-1,keepdims=True)
        k /= np.linalg.norm(k,axis=-1,keepdims=True)
        gates = -rng.uniform(0,0.3,size=q.shape).astype(dtype)
        beta = rng.uniform(0,1,size=(3,tokens)).astype(dtype)
        state = rng.normal(size=(3,5,5)).astype(dtype)
        return q,k,v,gates,beta,state

    def tiled_chunk(self, values, chunk_size, head_tile, value_tile):
        q,k,v,gates,beta,initial = values
        h,t,key_dim = q.shape
        output, final = np.empty_like(v), np.empty_like(initial)
        for head in range(0,h,head_tile):
            hs = slice(head,head+head_tile)
            for column in range(0,v.shape[-1],value_tile):
                vs = slice(column,column+value_tile)
                state = initial[hs,:,vs].copy()
                for start in range(0,t,chunk_size):
                    ts = slice(start,start+chunk_size)
                    program = ChunkKDAProgram(q[hs].shape[0],key_dim,state.shape[-1],q[hs,ts].shape[1])
                    result,state = ChunkKDAInterpreter(program).run(q[hs,ts],k[hs,ts],v[hs,ts,vs],
                                                                  gates[hs,ts],beta[hs,ts],state)
                    output[hs,ts,vs] = result
                final[hs,:,vs] = state
        return output, final

    def test_chunk_graph_matches_recurrence_across_tiles_and_partial_chunks(self):
        for tokens in (1,5,17,65):
            values = self.inputs(tokens)
            expected = RecurrentKDAReference.run(*values)
            for chunk,heads,columns in ((1,3,5),(4,2,2),(16,1,5),(64,2,3)):
                with self.subTest(tokens=tokens,chunk=chunk,heads=heads,columns=columns):
                    result = self.tiled_chunk(values,chunk,heads,columns)
                    for actual,reference in zip(result,expected):
                        np.testing.assert_allclose(actual,reference,rtol=1e-11,atol=1e-11)

    def test_fp32_chunk_algebra_and_prefill_decode_continuation(self):
        q,k,v,gates,beta,initial = self.inputs(19,np.float32)
        initial.fill(0)
        expected, state = RecurrentKDAReference.run(q,k,v,gates,beta,initial)
        chunk_output, chunk_state = self.tiled_chunk((q,k,v,gates,beta,initial),8,2,3)
        np.testing.assert_allclose(chunk_output,expected,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(chunk_state,state,rtol=2e-5,atol=2e-6)
        prefix = tuple(x[:,:13] for x in (q,k,v,gates,beta))+(initial,)
        out, cached = self.tiled_chunk(prefix,8,2,3)
        remaining, cached = RecurrentKDAReference.run(q[:,13:],k[:,13:],v[:,13:],gates[:,13:],beta[:,13:],cached)
        np.testing.assert_allclose(np.concatenate((out,remaining),axis=1),expected,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(cached,state,rtol=2e-5,atol=2e-6)

    def test_zero_update_preserves_decayed_state_and_causal_output(self):
        q,k,v,gates,beta,initial = self.inputs(7)
        beta.fill(0)
        result,state = self.tiled_chunk((q,k,v,gates,beta,initial),3,2,2)
        expected_state = initial*np.exp(gates.sum(axis=1))[...,None]
        np.testing.assert_allclose(state,expected_state,rtol=1e-12,atol=1e-12)
        reference = RecurrentKDAReference.run(q,k,v,gates,beta,initial)[0]
        np.testing.assert_allclose(result,reference,rtol=1e-12,atol=1e-12)

    def test_independent_chunk_costs_and_factory(self):
        program = BaseKDAProgram.create_from_name('chunk',heads=2,key_dim=3,value_dim=2,tokens=4)
        self.assertIsInstance(program,ChunkKDAProgram)
        self.assertEqual(program.operation_counts()[0],340)
        self.assertEqual(program.state_bytes,48)
        self.assertEqual(program.graph.tensors['lower'].nbytes,128)
        self.assertEqual(program.graph.tensors['wu'].nbytes,160)
        order = program.graph.export()['schedule']
        self.assertLess(order.index('triangular_solve'),order.index('state_residual'))
        self.assertLess(order.index('output'),order.index('state_update'))

    def test_resident_tile_state_written_once_and_decode_read_once(self):
        cfg = KDAConfig(embedding_dim=8,num_heads=3,head_dim=5,algorithm='chunk',
                        chunk_size=4,head_tile_size=2,value_tile_size=3)
        for stage,tokens in (('prefill',11),('decode',1)):
            schedule = KDACoreSchedule(cfg,2,tokens,stage)
            phases = schedule.lower(AcceleratorRates(100,20,64,100000))
            state_write = sum(p.memory_bytes for p in phases if p.name.endswith('state_write'))
            state_read = sum(p.memory_bytes for p in phases if p.name.endswith('state_read'))
            self.assertEqual(state_write,2*3*5*5*4)
            self.assertEqual(state_read,state_write if stage=='decode' else 0)
            if stage=='prefill':
                self.assertTrue(any(p.iterations==2 for p in phases))
                self.assertTrue(any(p.name.endswith('chunk3') for p in phases))
            self.assertEqual(schedule.algorithm,'chunk' if stage=='prefill' else 'recurrent')

    def test_sram_boundary_switches_to_causal_spill(self):
        cfg = KDAConfig(embedding_dim=8,num_heads=1,head_dim=5,algorithm='chunk',chunk_size=4)
        schedule = KDACoreSchedule(cfg,1,9,'prefill')
        capacity = schedule.peak_scratch_bytes()
        resident = schedule.lower(AcceleratorRates(100,20,64,capacity))
        spilled = schedule.lower(AcceleratorRates(100,20,64,capacity-1))
        traffic = lambda phases: sum(p.memory_bytes*p.iterations for p in phases)
        self.assertGreater(traffic(spilled),traffic(resident))
        self.assertFalse(any(p.name.endswith('state_write') for p in spilled))
        writes = [i for i,p in enumerate(spilled) if p.name.endswith(':state_update')]
        self.assertEqual(len(writes),3)
        first = next(p for p in spilled if ':t0:state_residual' in p.name)
        next_chunk = next(p for p in spilled if ':t4:state_residual' in p.name)
        self.assertEqual(next_chunk.read_bytes-first.read_bytes,100)
        self.assertLess(writes[0],spilled.index(next_chunk))

    def test_value_tiling_reduces_residency_and_accounts_for_recomputation(self):
        cfg = KDAConfig(embedding_dim=8,num_heads=3,head_dim=8,algorithm='chunk',chunk_size=4)
        full = KDACoreSchedule(cfg,1,9,'prefill')
        tiled = KDACoreSchedule(replace(cfg,value_tile_size=2),1,9,'prefill')
        self.assertLess(tiled.peak_scratch_bytes(),full.peak_scratch_bytes())
        self.assertGreater(tiled.operation_counts()[0],full.operation_counts()[0])
        full_phases = full.lower(AcceleratorRates(100,20,64,100000))
        tiled_phases = tiled.lower(AcceleratorRates(100,20,64,100000))
        self.assertGreater(sum(p.memory_bytes*p.iterations for p in tiled_phases),
                           sum(p.memory_bytes*p.iterations for p in full_phases))

    def test_outer_graph_stages_core_inputs_and_retains_full_state_capacity(self):
        cfg = KDAConfig(embedding_dim=8,num_heads=3,head_dim=5,algorithm='chunk',chunk_size=4)
        graph = KDAProfile().graph(**dict(vars(cfg),bs=9,batch_size=2,L_seq=9,stage='prefill'))
        self.assertEqual(graph.tensors['matrix_after'].nbytes,2*3*5*5*4)
        self.assertEqual(sum(t.nbytes for t in graph.tensors.values() if t.storage=='weight'),cfg.parameter_count)
        phases = graph.lower(AcceleratorRates(100,20,64,100000))
        producer = next(p for p in phases if p.name=='q_normalize')
        self.assertEqual(producer.write_bytes,2*9*3*5)
        self.assertEqual(graph.schedules['delta_core'].operation_counts()[0],
                         next(n.macs for n in graph.nodes if n.name=='delta_core'))

    def test_invalid_algorithm_and_tile_configurations_fail(self):
        for arguments in ({'algorithm':'invalid'},{'chunk_size':0},{'head_tile_size':0},
                          {'value_tile_size':-1},{'value_tile_size':129},
                          {'algorithm':'chunk','state_mapping':'whole'}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                KDAConfig(**arguments)


if __name__ == '__main__':
    unittest.main()
