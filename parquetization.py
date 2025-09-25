import duckdb

duckdb.sql("""
COPY (
  SELECT niq_be, prc_highprc_252d, niq_at, mispricing_perf, ni_be, ebit_bev, me, ret_12_1,
    ni_me, corr_1260d, op_atl1, ope_be, ocf_at, op_at, qmj_prof, ope_bel1, prc, eqnpo_12m, qmj, seas_1_1na, age, 
    ebitda_mev, ocf_me, ret_9_1, fcf_me, f_score, qmj_safety, ret_12_7, cop_atl1, cop_at, ret_6_1, dolvol_126d,
    intrinsic_value, niq_be_chg1, div12m_me, ni_inc8q, gp_at, be_gr1a, niq_at_chg1, resff3_12_1, id, date, ret_eom, gvkey, iid,
    excntry, stock_ret, year, month, char_date, char_eom
  FROM read_csv_auto('ret_sample.csv', HEADER=TRUE)
)
TO 'ret_parquets'
(FORMAT 'parquet', PARTITION_BY (year, month));
""")
