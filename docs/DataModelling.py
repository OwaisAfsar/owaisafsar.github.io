import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

each_value = []
report_data = []
column_names = ["CF", "DeveloperRoles", "MS", "CMS", "CMSbyMS"]


def research_question():
    main_data = pd.read_csv('CSVFiles/ms-data.csv', index_col=0)

    dev_data = main_data[['ms_id', 't_dev', 'has_conflict', 'l_core',
                          'l_peripheral', 'r_core', 'r_peripheral', 'top', 'occ', 'l_top', 'l_occ', 'r_top', 'r_occ']]
    chunk_data = main_data[['ms_id', 't_ch', 'has_conflict', 'l_core'
                            , 'l_peripheral', 'r_core', 'r_peripheral', 'top', 'occ', 'l_top', 'l_occ', 'r_top',
                            'r_occ']]
    chunks_branch_data = main_data[
        ['ms_id', 't_ch', 'core', 'peripheral', 'top', 'occ', 'l_core', 'l_peripheral', 'r_core', 'r_peripheral',
         'l_top', 'l_occ', 'r_top', 'r_occ', 'core_and_top', 'core_and_occ', 'peripheral_and_top', 'peripheral_and_occ',
         'l_core_and_top', 'l_core_and_occ', 'l_peripheral_and_top', 'l_peripheral_and_occ', 'r_core_and_top',
         'r_core_and_occ', 'r_peripheral_and_top', 'r_peripheral_and_occ', 'has_conflict']]
    dev_branch_data = main_data[
        ['ms_id', 't_dev', 'core', 'peripheral', 'top', 'occ', 'l_core', 'l_peripheral', 'r_core', 'r_peripheral',
         'l_top', 'l_occ', 'r_top', 'r_occ', 'core_and_top', 'core_and_occ', 'peripheral_and_top', 'peripheral_and_occ',
         'l_core_and_top', 'l_core_and_occ', 'l_peripheral_and_top', 'l_peripheral_and_occ', 'r_core_and_top',
         'r_core_and_occ', 'r_peripheral_and_top', 'r_peripheral_and_occ', 'has_conflict']]

    # CHECK RELATION OF THE VARIABLES

    # Why categorised developers <=5, between 5 and 10, >10
    dev_less_5 = dev_data.loc[dev_data['t_dev'] <= 5]
    dev_bt_5_10 = dev_data.loc[(dev_data['t_dev'] > 5) & (dev_data['t_dev'] < 11)]
    dev_gt_10 = dev_data.loc[dev_data['t_dev'] > 10]

    cn_dev_less_5 = dev_less_5.loc[dev_less_5['has_conflict'] == 1]
    cn_dev_bt_5_10 = dev_bt_5_10.loc[dev_bt_5_10['has_conflict'] == 1]
    cn_dev_gt_10 = dev_gt_10.loc[dev_gt_10['has_conflict'] == 1]

    # Plot Developers Distribution
    developer_distribution(dev_less_5, dev_bt_5_10, dev_gt_10)
    developer_conflict_distribution(cn_dev_less_5, cn_dev_bt_5_10, cn_dev_gt_10)

    # Why categorised chunks <= 50, between 50 and 100, > 100
    chunks_less_50 = chunk_data.loc[chunk_data['t_ch'] <= 50]
    chunks_bt_50_100 = chunk_data.loc[(chunk_data['t_ch'] > 50) & (chunk_data['t_ch'] < 101)]
    chunks_gt_100 = chunk_data.loc[chunk_data['t_ch'] > 100]

    cn_chunks_less_50 = chunks_less_50.loc[chunks_less_50['has_conflict'] == 1]
    cn_chunks_bt_50_100 = chunks_bt_50_100.loc[chunks_bt_50_100['has_conflict'] == 1]
    cn_chunks_gt_100 = chunks_gt_100.loc[chunks_gt_100['has_conflict'] == 1]

    # Plot Chunks Distribution
    chunks_distribution(chunks_less_50, chunks_bt_50_100, chunks_gt_100)
    chunks_conflict_distribution(cn_chunks_less_50, cn_chunks_bt_50_100, cn_chunks_gt_100)

    # RESEARCH QUESTIONS
    dev5 = dev_data.loc[dev_data['t_dev'] <= 5]
    dev10 = dev_data.loc[(dev_data['t_dev'] > 5) & (dev_data['t_dev'] < 11)]
    dev15 = dev_data.loc[dev_data['t_dev'] > 10]

    chunks50 = chunk_data.loc[chunk_data['t_ch'] <= 50]
    chunks100 = chunk_data.loc[(chunk_data['t_ch'] > 50) & (chunk_data['t_ch'] < 101)]
    chunks150 = chunk_data.loc[chunk_data['t_ch'] > 100]

    # Research Question #1
    # Top and occ. contributors at project-level (Research Question #1.1)
    top_Occasional_Project_level(dev5, dev10, dev15, chunks50, chunks100, chunks150, chunks_branch_data)

    # Top and occ. contributors at branch-level (Research Question #1.2)
    top_Occasional_Branch_level(dev5, dev10, dev15, chunks50, chunks100, chunks150, chunks_branch_data)

    # Research Question #2
    # Top and occ. contributors at project- and branch-level touch Target (Research Question #2.1)
    top_occ_dev_target(dev_branch_data, chunks_branch_data)

    # Top and occ. contributors at project- and branch-level touch Source (Research Question #2.2)
    top_occ_dev_chunks_source(dev_branch_data, chunks_branch_data)


# DEVELOPER DISTRIBUTION
def developer_distribution(dev_less_5, dev_bt_5_10, dev_gt_10):
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))

    category1 = dev_less_5.shape[0]
    category2 = dev_bt_5_10.shape[0]
    category3 = dev_gt_10.shape[0]

    data1 = category1
    data2 = category2
    data3 = category3

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d} \n {p:.2f}%'.format(p=pct, v=val)

        return my_autopct

    data = [data1, data2, data3]
    mccolors = ("#1ABC9C", "#76D7C4", "#A3E4D7")

    ax.pie(data, wedgeprops=dict(width=0.6), startangle=-170, colors=mccolors, autopct=make_autopct(data))
    ax.set_title("Share of #Developers", fontweight="bold")

    legend_x = .5
    legend_y = .05
    plt.legend(["Up to 5 developers in a merge scenario", "Between 5 to 10 developers in a merge scenario",
                "More than 10 developers in a merge scenario"], loc='upper center', bbox_to_anchor=(legend_x, legend_y))

    plt.savefig('CSVFiles/Developers_Distribution')
    plt.show()


# DEVELOPER CONFLICTS DISTRIBUTION
def developer_conflict_distribution(cn_dev_less_5, cn_dev_bt_5_10, cn_dev_gt_10):
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))

    category1 = cn_dev_less_5.shape[0]
    category2 = cn_dev_bt_5_10.shape[0]
    category3 = cn_dev_gt_10.shape[0]

    data1 = category1
    data2 = category2
    data3 = category3

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d} \n {p:.2f}%'.format(p=pct, v=val)

        return my_autopct

    data = [data1, data2, data3]
    mccolors = ("#8E44AD", "#D2B4DE", "#BB8FCE")

    ax.pie(data, wedgeprops=dict(width=0.6), startangle=-170, colors=mccolors, autopct=make_autopct(data))
    ax.set_title("Share of Merge Conflicts", fontweight="bold")

    legend_x = .5
    legend_y = .05
    plt.legend(["Up to 5 developers in a merge scenario", "Between 5 to 10 developers in a merge scenario",
                "More than 10 developers in a merge scenario"], loc='upper center', bbox_to_anchor=(legend_x, legend_y))

    plt.savefig('CSVFiles/Developers_Conflicts_Distribution')
    plt.show()


# CHUNKS DISTRIBUTION
def chunks_distribution(chunks_less_50, chunks_bt_50_100, chunks_gt_100):
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))

    category1 = chunks_less_50.shape[0]
    category2 = chunks_bt_50_100.shape[0]
    category3 = chunks_gt_100.shape[0]

    data1 = category1
    data2 = category2
    data3 = category3

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d} \n {p:.2f}%'.format(p=pct, v=val)

        return my_autopct

    data = [data1, data2, data3]
    mccolors = ("#1ABC9C", "#76D7C4", "#A3E4D7")

    ax.pie(data, wedgeprops=dict(width=0.6), startangle=-170, colors=mccolors, autopct=make_autopct(data))
    ax.set_title("Share of #Chunks", fontweight="bold")

    legend_x = .5
    legend_y = .05
    plt.legend(["At most 50 chunks of code changed by developers in a merge scenario",
                "At least 50 and at most 100 chunks of code changed by developers in a merge scenario",
                "More than 100 chunks of code changed by developers in a merge scenario"], loc='upper center',
               bbox_to_anchor=(legend_x, legend_y))

    plt.savefig('CSVFiles/Chunks_Distribution')
    plt.show()


# CHUNKS CONFLICTS DISTRIBUTION
def chunks_conflict_distribution(cn_chunks_less_50, cn_chunks_bt_50_100, cn_chunks_gt_100):
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))

    category1 = cn_chunks_less_50.shape[0]
    category2 = cn_chunks_bt_50_100.shape[0]
    category3 = cn_chunks_gt_100.shape[0]

    data1 = category1
    data2 = category2
    data3 = category3

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d} \n {p:.2f}%'.format(p=pct, v=val)

        return my_autopct

    data = [data1, data2, data3]
    mccolors = ("#8E44AD", "#D2B4DE", "#BB8FCE")

    ax.pie(data, wedgeprops=dict(width=0.6), startangle=-170, colors=mccolors, autopct=make_autopct(data))
    ax.set_title("Share of Merge Conflicts", fontweight="bold")

    legend_x = .5
    legend_y = .05
    plt.legend(["At most 50 chunks of code changed by developers in a merge scenario",
                "At least 50 and at most 100 chunks of code changed by developers in a merge scenario",
                "More than 100 chunks of code changed by developers in a merge scenario"], loc='upper center',
               bbox_to_anchor=(legend_x, legend_y))

    plt.savefig('CSVFiles/Chunks_Conflicts_Distribution')
    plt.show()


# (Research Question #1.1)
def top_Occasional_Project_level(dev5, dev10, dev15, chunks50, chunks100, chunks150, chunks_branch_data):
    report_data.clear()

    cf = 'General'
    top_p = chunks_branch_data.loc[chunks_branch_data['core'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'top_p', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    occ_p = chunks_branch_data.loc[chunks_branch_data['peripheral'] > 0]
    occ_p_count = occ_p['has_conflict'].count()
    occ_p_conflict = occ_p['has_conflict'].sum()
    print(cf, occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100)
    each_value = [cf, 'occ_p', occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Branch'
    top_p_target = chunks_branch_data.loc[chunks_branch_data['l_core'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_p_target', top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks_branch_data.loc[chunks_branch_data['l_peripheral'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_p_target', occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks_branch_data.loc[chunks_branch_data['r_core'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_p_source', top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks_branch_data.loc[chunks_branch_data['r_peripheral'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source', occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Developers'
    # Developers <=5
    top_p_target = dev5.loc[dev5['l_core'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_p_target_dev<=5', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = dev5.loc[dev5['l_peripheral'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_p_target_dev<=5', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = dev5.loc[dev5['r_core'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_p_source_dev<=5', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = dev5.loc[dev5['r_peripheral'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source_dev<=5', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 5 & <= 10
    top_p_target = dev10.loc[dev10['l_core'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_p_target_(dev>5 & dev<=10)', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = dev10.loc[dev10['l_peripheral'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_p_target_(dev>5 & dev<=10)', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = dev10.loc[dev10['r_core'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_p_source_(dev>5 & dev<=10)', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = dev10.loc[dev10['r_peripheral'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source_(dev>5 & dev<=10)', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 10
    top_p_target = dev15.loc[dev15['l_core'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_p_target_dev>10', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = dev15.loc[dev15['l_peripheral'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_p_target_dev>10', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = dev15.loc[dev15['r_core'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_p_source_dev>10', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = dev15.loc[dev15['r_peripheral'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source_dev>10', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Chunks'
    # Chunks <=50
    top_p_target = chunks50.loc[chunks50['l_core'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_p_target_Chunks<=50', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks50.loc[chunks50['l_peripheral'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_p_target_Chunks<=50', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks50.loc[chunks50['r_core'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_p_source_Chunks<=50', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks50.loc[chunks50['r_peripheral'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source_Chunks<=50', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 50 & <= 100
    top_p_target = chunks100.loc[chunks100['l_core'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_p_target_(chunks>50 & chunks<=100)', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks100.loc[chunks100['l_peripheral'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_p_target_(chunks>50 & chunks<=100)', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks100.loc[chunks100['r_core'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_p_source_(chunks>50 & chunks<=100)', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks100.loc[chunks100['r_peripheral'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source_(chunks>50 & chunks<=100)', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 100
    top_p_target = chunks150.loc[chunks150['l_core'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_p_target_chunks>100', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks150.loc[chunks150['l_peripheral'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_p_target_chunks>100', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks150.loc[chunks150['r_core'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_p_source_chunks>100', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks150.loc[chunks150['r_peripheral'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source_chunks>100', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    df = pd.DataFrame(report_data, columns=column_names)
    df.to_csv('CSVFiles/RQ1.1_Results.csv')


# (Research Question #1.2)
def top_Occasional_Branch_level(dev5, dev10, dev15, chunks50, chunks100, chunks150, chunks_branch_data):
    report_data.clear()

    cf = 'General'
    top_b = chunks_branch_data.loc[chunks_branch_data['top'] > 0]
    top_b_count = top_b['has_conflict'].count()
    top_b_conflict = top_b['has_conflict'].sum()
    print(cf, top_b_count, top_b_conflict, round(top_b_conflict / top_b_count, 3) * 100)
    each_value = [cf, 'top_b', top_b_count, top_b_conflict, round(top_b_conflict / top_b_count, 3) * 100]
    report_data.append(each_value)

    occ_b = chunks_branch_data.loc[chunks_branch_data['occ'] > 0]
    occ_b_count = occ_b['has_conflict'].count()
    occ_b_conflict = occ_b['has_conflict'].sum()
    print(cf, occ_b_count, occ_b_conflict, round(occ_b_conflict / occ_b_count, 3) * 100)
    each_value = [cf, 'occ_p', occ_b_count, occ_b_conflict, round(occ_b_conflict / occ_b_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Branch'
    top_p_target = chunks_branch_data.loc[chunks_branch_data['l_top'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_b_target', top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks_branch_data.loc[chunks_branch_data['l_occ'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_b_target', occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks_branch_data.loc[chunks_branch_data['r_top'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_b_source', top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks_branch_data.loc[chunks_branch_data['r_occ'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_b_source', occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Developers'
    # Developers <=5
    top_p_target = dev5.loc[dev5['l_top'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_b_target_dev<=5', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = dev5.loc[dev5['l_occ'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_b_target_dev<=5', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = dev5.loc[dev5['r_top'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_b_source_dev<=5', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = dev5.loc[dev5['r_occ'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_p_source_dev<=5', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 5 & <= 10
    top_p_target = dev10.loc[dev10['l_top'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_b_target_(dev>5 & dev<=10)', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = dev10.loc[dev10['l_occ'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_b_target_(dev>5 & dev<=10)', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = dev10.loc[dev10['r_top'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_b_source_(dev>5 & dev<=10)', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = dev10.loc[dev10['r_occ'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_b_source_(dev>5 & dev<=10)', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 10
    top_p_target = dev15.loc[dev15['l_top'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_b_target_dev>10', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = dev15.loc[dev15['l_occ'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_b_target_dev>10', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = dev15.loc[dev15['r_top'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_b_source_dev>10', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = dev15.loc[dev15['r_occ'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_b_source_dev>10', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Chunks'
    # Chunks <=50
    top_p_target = chunks50.loc[chunks50['l_top'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_b_target_Chunks<=50', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks50.loc[chunks50['l_occ'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_b_target_Chunks<=50', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks50.loc[chunks50['r_top'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_b_source_Chunks<=50', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks50.loc[chunks50['r_occ'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_b_source_Chunks<=50', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 50 & <= 100
    top_p_target = chunks100.loc[chunks100['l_top'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_b_target_(chunks>50 & chunks<=100)', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks100.loc[chunks100['l_occ'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_b_target_(chunks>50 & chunks<=100)', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks100.loc[chunks100['r_top'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_b_source_(chunks>50 & chunks<=100)', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks100.loc[chunks100['r_occ'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_b_source_(chunks>50 & chunks<=100)', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 100
    top_p_target = chunks150.loc[chunks150['l_top'] > 0]
    top_p_target_count = top_p_target['has_conflict'].count()
    top_p_target_conflict = top_p_target['has_conflict'].sum()
    print(cf, top_p_target_count, top_p_target_conflict, round(top_p_target_conflict / top_p_target_count, 3) * 100)
    each_value = [cf, 'top_b_target_chunks>100', top_p_target_count, top_p_target_conflict,
                  round(top_p_target_conflict / top_p_target_count, 3) * 100]
    report_data.append(each_value)

    occ_p_target = chunks150.loc[chunks150['l_occ'] > 0]
    occ_p_target_count = occ_p_target['has_conflict'].count()
    occ_p_target_conflict = occ_p_target['has_conflict'].sum()
    print(cf, occ_p_target_count, occ_p_target_conflict, round(occ_p_target_conflict / occ_p_target_count, 3) * 100)
    each_value = [cf, 'occ_b_target_chunks>100', occ_p_target_count, occ_p_target_conflict,
                  round(occ_p_target_conflict / occ_p_target_count, 3) * 100]
    report_data.append(each_value)

    top_p_source = chunks150.loc[chunks150['r_top'] > 0]
    top_p_source_count = top_p_source['has_conflict'].count()
    top_p_source_conflict = top_p_source['has_conflict'].sum()
    print(cf, top_p_source_count, top_p_source_conflict, round(top_p_source_conflict / top_p_source_count, 3) * 100)
    each_value = [cf, 'top_b_source_chunks>100', top_p_source_count, top_p_source_conflict,
                  round(top_p_source_conflict / top_p_source_count, 3) * 100]
    report_data.append(each_value)

    occ_p_source = chunks150.loc[chunks150['r_occ'] > 0]
    occ_p_source_count = occ_p_source['has_conflict'].count()
    occ_p_source_conflict = occ_p_source['has_conflict'].sum()
    print(cf, occ_p_source_count, occ_p_source_conflict, round(occ_p_source_conflict / occ_p_source_count, 3) * 100)
    each_value = [cf, 'occ_b_source_chunks>100', occ_p_source_count, occ_p_source_conflict,
                  round(occ_p_source_conflict / occ_p_source_count, 3) * 100]
    report_data.append(each_value)

    df = pd.DataFrame(report_data, columns=column_names)
    df.to_csv('CSVFiles/RQ1.2_Results.csv')


# (Research Question #2.1)
def top_occ_dev_target(dev_branch_data, chunks_branch_data):
    report_data.clear()

    cf = 'General'
    top_p = dev_branch_data.loc[dev_branch_data['core_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'top_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    occ_p = dev_branch_data.loc[dev_branch_data['core_and_occ'] > 0]
    occ_p_count = occ_p['has_conflict'].count()
    occ_p_conflict = occ_p['has_conflict'].sum()
    print(cf, occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100)
    each_value = [cf, 'top_p o occ_p', occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100]
    report_data.append(each_value)

    top_p = dev_branch_data.loc[dev_branch_data['peripheral_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'occ_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    variable = dev_branch_data.loc[dev_branch_data['peripheral_and_occ'] > 0]
    variable_count = variable['has_conflict'].count()
    variable_conflict = variable['has_conflict'].sum()
    print(cf, variable_count, variable_conflict, round(variable_conflict / variable_count, 3) * 100)
    each_value = [cf, 'occ_p o occ_b', variable_count, variable_conflict, round(variable_conflict / variable_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Branch'
    top_p = dev_branch_data.loc[dev_branch_data['l_core_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'top_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    occ_p = dev_branch_data.loc[dev_branch_data['l_core_and_occ'] > 0]
    occ_p_count = occ_p['has_conflict'].count()
    occ_p_conflict = occ_p['has_conflict'].sum()
    print(cf, occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100)
    each_value = [cf, 'top_p o occ_p', occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100]
    report_data.append(each_value)

    top_p = dev_branch_data.loc[dev_branch_data['l_peripheral_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'occ_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    variable = dev_branch_data.loc[dev_branch_data['l_peripheral_and_occ'] > 0]
    variable_count = variable['has_conflict'].count()
    variable_conflict = variable['has_conflict'].sum()
    print(cf, variable_count, variable_conflict, round(variable_conflict / variable_count, 3) * 100)
    each_value = [cf, 'occ_p o occ_b', variable_count, variable_conflict,
                  round(variable_conflict / variable_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Developers'
    # Developers <= 5
    top_p_top_b = dev_branch_data.loc[(dev_branch_data['l_core_and_top'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = dev_branch_data.loc[(dev_branch_data['l_core_and_occ'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = dev_branch_data.loc[
        (dev_branch_data['l_peripheral_and_top'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = dev_branch_data.loc[
        (dev_branch_data['l_peripheral_and_occ'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o target o Dev<=5: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o target o Dev<=5', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o target o Dev<=5: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o target o Dev<=5', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o target o Dev<=5: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o target o Dev<=5', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o target o Dev<=5: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o target o Dev<=5', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 5 & <= 10
    top_p_top_b = dev_branch_data.loc[(dev_branch_data['l_core_and_top'] > 0) & (dev_branch_data['t_dev'] > 5) & (
                dev_branch_data['t_dev'] <= 10)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = dev_branch_data.loc[(dev_branch_data['l_core_and_occ'] > 0) & (dev_branch_data['t_dev'] > 5) & (
                dev_branch_data['t_dev'] <= 10)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = dev_branch_data.loc[
        (dev_branch_data['l_peripheral_and_top'] > 0) & (dev_branch_data['t_dev'] > 5) & (
                    dev_branch_data['t_dev'] <= 10)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = dev_branch_data.loc[
        (dev_branch_data['l_peripheral_and_occ'] > 0) & (dev_branch_data['t_dev'] > 5) & (
                    dev_branch_data['t_dev'] <= 10)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o target o (dev>5 & dev<=10): ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o target o (dev>5 & dev<=10)', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o target o (dev>5 & dev<=10): ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o target o (dev>5 & dev<=10)', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o target o (dev>5 & dev<=10): ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o target o (dev>5 & dev<=10)', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o target o (dev>5 & dev<=10): ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o target o (dev>5 & dev<=10)', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 10
    top_p_top_b = dev_branch_data.loc[(dev_branch_data['l_core_and_top'] > 0) & (dev_branch_data['t_dev'] > 10)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = dev_branch_data.loc[(dev_branch_data['l_core_and_occ'] > 0) & (dev_branch_data['t_dev'] > 10)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = dev_branch_data.loc[
        (dev_branch_data['l_peripheral_and_top'] > 0) & (dev_branch_data['t_dev'] > 10)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = dev_branch_data.loc[
        (dev_branch_data['l_peripheral_and_occ'] > 0) & (dev_branch_data['t_dev'] > 10)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o target o Dev>10: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o target o Dev>10', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o target o Dev>10: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o target o Dev>10', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o target o Dev>10: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o target o Dev>10', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o target o Dev>10: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o target o Dev>10', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Chunks'
    # Chunks <= 50
    top_p_top_b = chunks_branch_data.loc[(chunks_branch_data['l_core_and_top'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = chunks_branch_data.loc[(chunks_branch_data['l_core_and_occ'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['l_peripheral_and_top'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['l_peripheral_and_occ'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o target o Chunks<=50: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o target o Chunks<=50', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o target o Chunks<=50: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o target o Chunks<=50', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o target o Chunks<=50: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o target o Chunks<=50', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o target o Chunks<=50: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o target o Chunks<=50', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 50 & <= 100
    top_p_top_b = chunks_branch_data.loc[(chunks_branch_data['l_core_and_top'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
            chunks_branch_data['t_ch'] <= 100)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = chunks_branch_data.loc[(chunks_branch_data['l_core_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
            chunks_branch_data['t_ch'] <= 100)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['l_peripheral_and_top'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
                chunks_branch_data['t_ch'] <= 100)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['l_peripheral_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
                chunks_branch_data['t_ch'] <= 100)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o target o (chunks>50 & chunks<=100): ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o target o (chunks>50 & chunks<=100)', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o target o (chunks>50 & chunks<=100): ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o target o (chunks>50 & chunks<=100)', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o target o (chunks>50 & chunks<=100): ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o target o (chunks>50 & chunks<=100)', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o target o (chunks>50 & chunks<=100): ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o target o (chunks>50 & chunks<=100)', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 100
    top_p_top_b = chunks_branch_data.loc[(chunks_branch_data['l_core_and_top'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = chunks_branch_data.loc[(chunks_branch_data['l_core_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['l_peripheral_and_top'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['l_peripheral_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o target o Chunks>100: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o target o Chunks>100', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o target o Chunks>100: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o target o Chunks>100', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o target o Chunks>100: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o target o Chunks>100', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o target o Chunks>100: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o target o Chunks>100', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    df = pd.DataFrame(report_data, columns=column_names)
    df.to_csv('CSVFiles/RQ2.1_Results.csv')


# (Research Question #2.2)
def top_occ_dev_chunks_source(dev_branch_data, chunks_branch_data):
    report_data.clear()

    print('Research Question 2.2')
    cf = 'General'
    top_p = dev_branch_data.loc[dev_branch_data['core_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'top_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    occ_p = dev_branch_data.loc[dev_branch_data['core_and_occ'] > 0]
    occ_p_count = occ_p['has_conflict'].count()
    occ_p_conflict = occ_p['has_conflict'].sum()
    print(cf, occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100)
    each_value = [cf, 'top_p o occ_p', occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100]
    report_data.append(each_value)

    top_p = dev_branch_data.loc[dev_branch_data['peripheral_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'occ_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    variable = dev_branch_data.loc[dev_branch_data['peripheral_and_occ'] > 0]
    variable_count = variable['has_conflict'].count()
    variable_conflict = variable['has_conflict'].sum()
    print(cf, variable_count, variable_conflict, round(variable_conflict / variable_count, 3) * 100)
    each_value = [cf, 'occ_p o occ_b', variable_count, variable_conflict,
                  round(variable_conflict / variable_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Branch'
    top_p = dev_branch_data.loc[dev_branch_data['r_core_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'top_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    occ_p = dev_branch_data.loc[dev_branch_data['r_core_and_occ'] > 0]
    occ_p_count = occ_p['has_conflict'].count()
    occ_p_conflict = occ_p['has_conflict'].sum()
    print(cf, occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100)
    each_value = [cf, 'top_p o occ_p', occ_p_count, occ_p_conflict, round(occ_p_conflict / occ_p_count, 3) * 100]
    report_data.append(each_value)

    top_p = dev_branch_data.loc[dev_branch_data['r_peripheral_and_top'] > 0]
    top_p_count = top_p['has_conflict'].count()
    top_p_conflict = top_p['has_conflict'].sum()
    print(cf, top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100)
    each_value = [cf, 'occ_p o top_b', top_p_count, top_p_conflict, round(top_p_conflict / top_p_count, 3) * 100]
    report_data.append(each_value)

    variable = dev_branch_data.loc[dev_branch_data['r_peripheral_and_occ'] > 0]
    variable_count = variable['has_conflict'].count()
    variable_conflict = variable['has_conflict'].sum()
    print(cf, variable_count, variable_conflict, round(variable_conflict / variable_count, 3) * 100)
    each_value = [cf, 'occ_p o occ_b', variable_count, variable_conflict,
                  round(variable_conflict / variable_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Developers'
    # Dev <= 5
    top_p_top_b = dev_branch_data.loc[(dev_branch_data['r_core_and_top'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = dev_branch_data.loc[(dev_branch_data['r_core_and_occ'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = dev_branch_data.loc[
        (dev_branch_data['r_peripheral_and_top'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = dev_branch_data.loc[
        (dev_branch_data['r_peripheral_and_occ'] > 0) & (dev_branch_data['t_dev'] <= 5)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o source o Dev<=5: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o source o Dev<=5', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o source o Dev<=5: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o source o Dev<=5', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o source o Dev<=5: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o source o Dev<=5', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o source o Dev<=5: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o source o Dev<=5', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 5 & <= 10
    top_p_top_b = dev_branch_data.loc[(dev_branch_data['r_core_and_top'] > 0) & (dev_branch_data['t_dev'] > 5) & (
            dev_branch_data['t_dev'] <= 10)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = dev_branch_data.loc[(dev_branch_data['r_core_and_occ'] > 0) & (dev_branch_data['t_dev'] > 5) & (
            dev_branch_data['t_dev'] <= 10)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = dev_branch_data.loc[
        (dev_branch_data['r_peripheral_and_top'] > 0) & (dev_branch_data['t_dev'] > 5) & (
                dev_branch_data['t_dev'] <= 10)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = dev_branch_data.loc[
        (dev_branch_data['r_peripheral_and_occ'] > 0) & (dev_branch_data['t_dev'] > 5) & (
                dev_branch_data['t_dev'] <= 10)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o source o (dev>5 & dev<=10): ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o source o (dev>5 & dev<=10)', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o source o (dev>5 & dev<=10): ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o source o (dev>5 & dev<=10)', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o source o (dev>5 & dev<=10): ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o source o (dev>5 & dev<=10)', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o source o (dev>5 & dev<=10): ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o source o (dev>5 & dev<=10)', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Developers > 10
    top_p_top_b = dev_branch_data.loc[(dev_branch_data['r_core_and_top'] > 0) & (dev_branch_data['t_dev'] > 10)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = dev_branch_data.loc[(dev_branch_data['r_core_and_occ'] > 0) & (dev_branch_data['t_dev'] > 10)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = dev_branch_data.loc[
        (dev_branch_data['r_peripheral_and_top'] > 0) & (dev_branch_data['t_dev'] > 10)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = dev_branch_data.loc[
        (dev_branch_data['r_peripheral_and_occ'] > 0) & (dev_branch_data['t_dev'] > 10)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o source o Dev>10: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o source o Dev>10', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o source o Dev>10: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o source o Dev>10', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o source o Dev>10: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o source o Dev>10', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o source o Dev>10: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o source o Dev>10', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    cf = 'Chunks'
    # Chunks <= 50
    top_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['r_core_and_top'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['r_core_and_occ'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['r_peripheral_and_top'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['r_peripheral_and_occ'] > 0) & (chunks_branch_data['t_ch'] <= 50)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o source o Chunks<=50: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o target o Chunks<=50', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o source o Chunks<=50: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o target o Chunks<=50', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o source o Chunks<=50: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o target o Chunks<=50', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o source o Chunks<=50: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o target o Chunks<=50', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 50 & <= 100
    top_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['r_core_and_top'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
                chunks_branch_data['t_ch'] <= 100)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['r_core_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
                chunks_branch_data['t_ch'] <= 100)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['r_peripheral_and_top'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
                chunks_branch_data['t_ch'] <= 100)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['r_peripheral_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 50) & (
                chunks_branch_data['t_ch'] <= 100)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o source o (chunks>50 & chunks<=100): ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o source o (chunks>50 & chunks<=100)', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o source o (chunks>50 & chunks<=100): ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o source o (chunks>50 & chunks<=100)', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o source o (chunks>50 & chunks<=100): ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o source o (chunks>50 & chunks<=100)', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o source o (chunks>50 & chunks<=100): ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o source o (chunks>50 & chunks<=100)', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    # Chunks > 100
    top_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['r_core_and_top'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    top_p_top_b_count = top_p_top_b['has_conflict'].count()
    top_p_top_b_conflict = top_p_top_b['has_conflict'].sum()

    top_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['r_core_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    top_p_occ_b_count = top_p_occ_b['has_conflict'].count()
    top_p_occ_b_conflict = top_p_occ_b['has_conflict'].sum()

    occ_p_top_b = chunks_branch_data.loc[
        (chunks_branch_data['r_peripheral_and_top'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    occ_p_top_b_count = occ_p_top_b['has_conflict'].count()
    occ_p_top_b_conflict = occ_p_top_b['has_conflict'].sum()

    occ_p_occ_b = chunks_branch_data.loc[
        (chunks_branch_data['r_peripheral_and_occ'] > 0) & (chunks_branch_data['t_ch'] > 100)]
    occ_p_occ_b_count = occ_p_occ_b['has_conflict'].count()
    occ_p_occ_b_conflict = occ_p_occ_b['has_conflict'].sum()

    print("Top_p o Top_b o source o Chunks>100: ", top_p_top_b_count, top_p_top_b_conflict,
          round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Top_b o source o Chunks>100', top_p_top_b_count, top_p_top_b_conflict,
                  round(top_p_top_b_conflict / top_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Top_p o Occ_b o source o Chunks>100: ", top_p_occ_b_count, top_p_occ_b_conflict,
          round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Top_p o Occ_b o source o Chunks>100', top_p_occ_b_count, top_p_occ_b_conflict,
                  round(top_p_occ_b_conflict / top_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Top_b o source o Chunks>100: ", occ_p_top_b_count, occ_p_top_b_conflict,
          round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Top_b o source o Chunks>100', occ_p_top_b_count, occ_p_top_b_conflict,
                  round(occ_p_top_b_conflict / occ_p_top_b_count, 3) * 100]
    report_data.append(each_value)

    print("Occ_p o Occ_b o source o Chunks>100: ", occ_p_occ_b_count, occ_p_occ_b_conflict,
          round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100)
    each_value = [cf, 'Occ_p o Occ_b o source o Chunks>100', occ_p_occ_b_count, occ_p_occ_b_conflict,
                  round(occ_p_occ_b_conflict / occ_p_occ_b_count, 3) * 100]
    report_data.append(each_value)

    df = pd.DataFrame(report_data, columns=column_names)
    df.to_csv('CSVFiles/RQ2.2_Results.csv')