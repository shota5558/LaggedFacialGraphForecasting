"""Hand-calculated checks for the new report estimands and legacy boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagged_facial_graph_forecasting.displacement_analysis import (
    CONDITIONS, POINTS, PROTOCOL, REGIONS, UNIT, band_gains, bootstrap_indices,
    centered_response, enrichment, summarize, validate_inputs,
)
from lagged_facial_graph_forecasting.metrics import displacement_mean_euclidean_error, DISPLACEMENT_EUCLIDEAN_NAME


def test_point_distance_precedes_point_and_time_average():
    truth = np.zeros((2, 2, 2))
    prediction = np.array([[[3,4],[-3,-4]], [[0,0],[0,2]]])
    # Opposite point directions must not cancel in a centroid; RMSE differs.
    assert displacement_mean_euclidean_error(truth, prediction) == 3
    with pytest.raises(ValueError, match="shape"):
        displacement_mean_euclidean_error(truth.reshape(2,4), prediction.reshape(2,4))
    prediction[0,0,0] = 0
    bad = prediction.astype(float); bad[0,0,0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        displacement_mean_euclidean_error(truth,bad)


def test_band_mean_precedes_subject_median_and_failure_is_not_partial_mean():
    rows=[]
    for s, values in enumerate(((100.,0.,0.), (0.,100.,0.), (0.,0.,100.))):
        for lag, gain in enumerate(values,1):
            rows.append(dict(condition="speaking",outer_fold=0,group_id=f"g{s}",subject_id=f"s{s}",
                             target="mouth",source="jaw",lag=lag,gain=gain,status="evaluable",support_sha256="a"*64))
    frame=pd.DataFrame(rows)
    result=band_gains(frame,30)
    assert result.gain.median() == pytest.approx(100/3)
    frame.loc[0,"status"]="failed";frame.loc[0,"gain"]=np.nan
    assert np.isnan(band_gains(frame,30).iloc[0].gain)


def test_group_bootstrap_keeps_pairs_fold_counts_and_variable_group_sizes():
    subjects=pd.DataFrame(dict(subject_id=["a","b","c","d"],group_id=["g1","g1","g2","g3"],outer_fold=[0,0,0,1]))
    draws=bootstrap_indices(subjects,50,9)
    np.testing.assert_array_equal(draws,bootstrap_indices(subjects,50,9))
    for row in draws:
        assert (row==0).sum() == (row==1).sum()
        assert (row==0).sum()+(row==2).sum()==2  # two sampled groups in fold 0
        assert (row==3).sum()==1
    frame=subjects.assign(condition="speaking",gain=[1.,2.,3.,4.])
    summary=summarize(frame,["condition"],"gain",subjects,draws).iloc[0]
    assert summary["median"]==2.5
    assert summary.ci_status=="fewer_than_10_groups" and np.isnan(summary.ci_low)


def test_centered_response_averages_lags_then_pairs():
    base=dict(condition="speaking",outer_fold=0,group_id="g1",subject_id="s1")
    rows=[]
    for source in ("jaw","left_cheek"):
        for lag in range(1,11):
            error=({5:0.,6:1.,7:4.}.get(lag,0.) if source=="jaw" else {6:6.}.get(lag,0.))
            rows.append(dict(base,target="mouth",source=source,lag=lag,cell_error=error,status="evaluable",support_sha256="a"*64))
    selected=pd.DataFrame([dict(condition="speaking",outer_fold=0,target="mouth",source=s,lag=l,selected=True)
                           for s,l in (("jaw",5),("jaw",6),("left_cheek",5),("jaw",1))])
    raw,subject,audit=centered_response(pd.DataFrame(rows),selected,10,1)
    assert subject.set_index("delta").loc[1,"response"]==4.  # ((1+3)/2 + 6)/2
    assert subject.set_index("delta").loc[0,"response"]==0.
    assert subject.n_edges.nunique()==1 and subject.n_pairs.nunique()==1
    assert audit.iloc[0].n_boundary==1
    assert raw.tau_star.min()>1


def test_enrichment_uses_mean_of_all_repeats_and_rejects_incomplete_grid():
    base=dict(condition="speaking",outer_fold=0,group_id="g1",subject_id="s1",target="mouth")
    names=["a","b","c","d"]
    cells=pd.DataFrame([dict(base,source=s,lag=1,gain=g,support_sha256="a"*64) for s,g in zip(names,[.004,.002,0.,.010])])
    selection=pd.DataFrame([dict(condition="speaking",outer_fold=0,target="mouth",source=s,lag=1,selected=i<2,available=True,scalar_count=2) for i,s in enumerate(names)])
    mappings=pd.DataFrame([dict(condition="speaking",outer_fold=0,target="mouth",repeat=r,membership=membership,membership_sha256="a"*64) for r,membership in enumerate(["a:1;c:1","b:1;c:1","b:1;d:1"])])
    values,repeats=enrichment(cells,selection,mappings,3)
    assert values.iloc[0].enrichment==pytest.approx(0.)  # .003 - mean(.002,.001,.006)
    assert len(repeats)==3
    with pytest.raises(ValueError,match="repeat grid"):
        enrichment(cells,selection,mappings.iloc[:2],3)
    selection["selected"]=False
    empty,_=enrichment(cells,selection,mappings,3)
    assert empty.iloc[0].status=="unevaluable_empty_selected" and np.isnan(empty.iloc[0].enrichment)


def test_input_contract_rejects_old_metric_missing_cells_and_mismatched_support():
    config=dict(protocol_id=PROTOCOL,is_synthetic=True,publication_ready=False,primary_metric=DISPLACEMENT_EUCLIDEAN_NAME,
                matched_repeats=1000,bootstrap_repeats=10000,lag_max=1)
    provenance=dict(protocol_id=PROTOCOL,is_synthetic=True)
    subject=dict(subject_id="MOCK_S01",group_id="MOCK_G01",outer_fold=0)
    subjects=pd.DataFrame([dict(subject,**provenance)])
    metrics=[];cells=[];selected=[]
    for condition in CONDITIONS:
        for target in REGIONS:
            for model in ("persistence","self","full","pcmci"):
                metrics.append(dict(subject,condition=condition,target=target,model=model,error=.02,status="evaluable",support_sha256="a"*64,
                                    metric_name=DISPLACEMENT_EUCLIDEAN_NAME,**provenance))
            for source,points in zip(REGIONS,POINTS):
                if source==target:continue
                cells.append(dict(subject,condition=condition,target=target,source=source,lag=1,self_error=.02,cell_error=.018,gain=.002,status="evaluable",support_sha256="a"*64,
                                  metric_name=DISPLACEMENT_EUCLIDEAN_NAME,**provenance))
                selected.append(dict(condition=condition,outer_fold=0,target=target,source=source,lag=1,selected=False,available=True,scalar_count=len(points)*2,**provenance))
    metrics,cells,selected=map(pd.DataFrame,(metrics,cells,selected))
    validate_inputs(config,subjects,metrics,cells,selected)
    with pytest.raises(ValueError,match="new-metric"):
        validate_inputs(dict(config,primary_metric="velocity_rmse"),subjects,metrics,cells,selected)
    with pytest.raises(ValueError,match="candidate cell"):
        validate_inputs(config,subjects,metrics,cells.iloc[1:],selected)
    cells.loc[0,"support_sha256"]="b"*64
    with pytest.raises(ValueError,match="common support"):
        validate_inputs(config,subjects,metrics,cells,selected)
