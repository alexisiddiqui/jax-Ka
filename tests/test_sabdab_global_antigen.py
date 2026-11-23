import sabdab_v7_antigen as antigen


def test_global_identity_does_not_turn_a_short_local_motif_into_full_homology():
    a=antigen.aligner()
    assert antigen.identity('AAAAAAAAAA'+'Q'*90,'AAAAAAAAAA',a)==.1
    assert antigen.identity('AAAAAAAAAA','AAAAAAAAAA',a)==1.


def test_published_protein_and_peptide_boundaries_are_distinct():
    sequences={'q':'A'*10,'p':'A'*4,'too_short':'A'*3,'pep':'A'*8,'short_pep':'A'*7}
    targets=[dict(sequence_key=k,antigen_type=t,components=['c'],reference_ids=['ref'])
             for k,t in [('p','PROTEIN'),('too_short','PROTEIN'),('pep','PEPTIDE'),('short_pep','PEPTIDE')]]
    antigen.initialise(sequences,targets)
    result=antigen.compare_chunk(['q'])
    assert result['pairs']==2
    assert {h['target'] for h in result['hits'][0]['hits']}=={'p','pep'}
    assert antigen.candidate_work(sequences,['q'],targets)['candidate_pairs']==2


def test_no_approximate_prefilter_is_needed_to_recover_an_exact_threshold_hit():
    # Four substitutions at the protein boundary: six mismatches and no gaps.
    score=antigen.identity('ACDEFGHIKL','ACDEQQQQQQ',antigen.aligner())
    assert score==.4
