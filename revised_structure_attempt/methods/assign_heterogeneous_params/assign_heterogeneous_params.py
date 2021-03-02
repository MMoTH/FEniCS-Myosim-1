import numpy as np
import numpy.random as r



## define heterogeneous parameters based on some rule
def assign_heterogeneous_params(sim_params,hs_params_template,hs_params_list,dolfin_functions,geo_options,no_of_int_points):

    # Going to directly go through hs_params_list and then dolfin_functions and check for heterogeneity
    # hs_params_template is the base copy of myosim parameters, loop through this
    seed = sim_params["rseed"][0]
    r.seed(seed)

    # create empty dictionary that will hold keys for heterogeneous hs parameters
    het_hs_dict = {}

    # fill het_hs_dict with any keys that are flagged as heterogeneous
    het_hs_dict = iterate_hs_keys(hs_params_template,het_hs_dict)

    # assign heterogeneous parameters based on the desired law
    hs_params_list = assign_hs_values(het_hs_dict,hs_params_list,no_of_int_points,geo_options) #geo_options will contain information for specific spatial variations

    # create empty dictionary to hold keys for heterogeneous dolfin functions
    het_dolfin_dict = {}

    # fill het_dolfin_dict with any keys that are flagged as heterogeneous
    #print "dolfin functions"
    #rint dolfin_functions
    het_dolfin_dict = iterate_dolfin_keys(dolfin_functions,het_dolfin_dict)

    # assign heterogeneous parametrs based on the desired law
    dolfin_functions = assign_dolfin_functions(dolfin_functions,het_dolfin_dict,no_of_int_points,geo_options)

    # Kurtis needs to update this
    #--------------------------------------------------------
    # For fiber simulations, ends need to not contract, and may have different
    # stiffness than the contractile tissue
    """if sim_params["simulation_geometry"][0] == "cylinder" or sim_params["simulation_geometry"][0] == "box_mesh" or sim_params["simulation_geometry"][0] == "gmesh_cylinder":

        end_marker_array = geo_options["end_marker_array"]

        fibrous_c  = geo_options["fibrous_c"]
        fibrous_c2 = geo_options["fibrous_c2"]
        fibrous_c3 = geo_options["fibrous_c3"]

        for jj in np.arange(no_of_int_points):
            #print "type" +str(end_marker_array[jj])

            if end_marker_array[jj] > 9.0 or end_marker_array[jj] < 1.0:
                hs_params_list[jj]["myofilament_parameters"]["k_3"][0] = 0.0
                passive_params_list[jj]["c"]  = fibrous_c[0]
                passive_params_list[jj]["c2"] = fibrous_c2[0]
                passive_params_list[jj]["c3"] = fibrous_c3[0]

                fcn_list[0].vector()[jj] = fibrous_c[0]
                fcn_list[1].vector()[jj] = fibrous_c2[0]
                fcn_list[2].vector()[jj] = fibrous_c3[0]
            else:

                #passive_params_list[jj]["c"] = passive_params_list[jj]["c"]
                #passive_params_list[jj]["c2"] = passive_params_list[jj]["c2"]
                #passive_params_list[jj]["c3"] = passive_params_list[jj]["c3"]
                fcn_list[0].vector()[jj] = passive_params_list[jj]["c"][0]
                fcn_list[1].vector()[jj] = passive_params_list[jj]["c2"][0]
                fcn_list[2].vector()[jj] = passive_params_list[jj]["c3"][0]

    else:

        for jj in np.arange(no_of_int_points):

            # assign them to be homogeneous until I put the option into the instruction files
            fcn_list[0].vector()[jj] = passive_params_list[0]["c"][0]
            fcn_list[1].vector()[jj] = passive_params_list[0]["c2"][0]
            fcn_list[2].vector()[jj] = passive_params_list[0]["c3"][0]"""


    return hs_params_list,dolfin_functions

def iterate_hs_keys(hs_template,het_hs_dict):

    for k, v in hs_template.items():

        if isinstance(v,dict):
            iterate_hs_keys(v,het_hs_dict)

        else:
            # got actual parameter value list, not another dictionary
            for j in v:
                if isinstance(j,dict):
                    if k == "cb_number_density":
                        print "pass up cb density here"
                    else:
                        check = j["heterogeneous"]
                        print "key is ", k
                        print "check is ", check
                        if (check=="true") or (check =="True"):
                            # this parameters should be homogeneous
                            temp_law = j["law"]
                            base_value = v[0] #first entry is base value
                            het_hs_dict[k]=[base_value,temp_law]
                            if temp_law == "gaussian":
                                if "width" in j:
                                    width = j["width"]
                                else:
                                    width = 0
                                het_hs_dict[k].append(width)
                            if temp_law == "percent_fibrosis":
                                if "percent" in j:
                                    percent = j["percent"]
                                else:
                                    percent = 0.33
                                het_hs_dict[k].append(percent)
                            if temp_law == "fiber_w_compliance":
                                if "fiber_value" in j:
                                    fiber_value = j["fiber_value"]
                                else:
                                    fiber_value = base_value
                                het_hs_dict[k].append(fiber_value)

    return het_hs_dict

def assign_hs_values(het_hs_dict,hs_params_list,no_of_int_points,geo_options):

    for k in het_hs_dict.keys():
        base_value = het_hs_dict[k][0]
        hetero_law = het_hs_dict[k][1]
        if hetero_law == "gaussian":
            hs_params_list = scalar_gaussian_law(hs_params_list,base_value,k,het_hs_dict[k][-1],no_of_int_points)

        if hetero_law == "percent_fibrosis":
            hs_params_list = scalar_fibrosis_law(hs_params_list,base_value,k,het_hs_dict[k][-1],no_of_int_points)

        else:
            print "instruction file law is",hetero_law
            print "invalid law. Please choose from `gaussian` or `percent_fibrosis`"

    return hs_params_list

def iterate_dolfin_keys(dolfin_functions,het_dolfin_dict):
    #print "dolfin function"
    #print dolfin_functions
    for k, v in dolfin_functions.items():
        print "key in dolfin keys is",k

        if isinstance(v,dict):
            iterate_dolfin_keys(v,het_dolfin_dict)

        else:
            # got actual parameter value list, not another dictionary
            for j in v:
                if isinstance(j,dict):
                    check = j["heterogeneous"]
                    if (check=="true") or (check=="True"):
                        #print "there is a hetero dict"
                        #print k
                        # this parameter should be homogeneous
                        temp_law = j["law"]
                        base_value = v[0] #first entry is base value
                        het_dolfin_dict[k]=[base_value,temp_law]
                        #print "het_dolfin_dict"
                        #print het_dolfin_dict
                        if temp_law == "gaussian":
                            if "width" in j:
                                width = j["width"]
                            else:
                                width = 1
                            het_dolfin_dict[k].append(width)
                        if temp_law == "percent_fibrosis":
                            if "percent" in j:
                                percent = j["percent"]
                            else:
                                percent = 0.33
                            het_dolfin_dict[k].append(percent)
                        if temp_law == "fiber_w_compliance":
                            if "fiber_value" in j:
                                fiber_value = j["fiber_value"]
                            else:
                                fiber_value = base_value
                            het_dolfin_dict[k].append(fiber_value)

    print "het_dolfin_dict is now "
    print het_dolfin_dict
    return het_dolfin_dict

def assign_dolfin_functions(dolfin_functions,het_dolfin_dict,no_of_int_points,geo_options):

    for k in het_dolfin_dict.keys():
        #print "het_dolfin_dict"
        #print k
        #print "assigning functions"
        #print het_dolfin_dict
        #print k
        base_value = het_dolfin_dict[k][0]
        hetero_law = het_dolfin_dict[k][1]

        if hetero_law == "gaussian":
            dolfin_functions = df_gaussian_law(dolfin_functions,base_value,k,het_dolfin_dict[k][-1],no_of_int_points)

        if hetero_law == "percent_fibrosis":
            dolfin_functions = df_fibrosis_law(dolfin_functions,base_value,k,het_dolfin_dict[k][-1],no_of_int_points)

        if hetero_law == "fiber_w_compliance":
            dolfin_functions = df_fiber_w_compliance_law(dolfin_functions,base_value,k,het_dolfin_dict[k][-1],no_of_int_points,geo_options)

    return dolfin_functions

def scalar_gaussian_law(hs_params_list,base_value,k,width,no_of_int_points):

    # generate random values for parameter k using gaussian distribution centered at base_value
    # with width specified by user
    values_array = r.normal(base_value,width,no_of_int_points)

    for jj in np.arange(no_of_int_points):
        # right now, assuming that only myofilmaent parameters change
        hs_params_list[jj]["myofilament_parameters"][k][0] = values_array[jj]

    return hs_params_list

def df_gaussian_law(dolfin_functions,base_value,k,width,no_of_int_points):

    values_array = r.normal(base_value,width,no_of_int_points)

    #print "gauss law"
    #print dolfin_functions["passive_params"][k]

    if k == "cb_number_density":
        dolfin_functions[k][-1].vector()[:] = values_array #last element in list is the initialized function
    else:
        dolfin_functions["passive_params"][k][-1].vector()[:] = values_array #last element in list is the initialized function

    return dolfin_functions

def scalar_fibrosis_law(hs_params_list,base_value,k,percent,no_of_int_points):

    sample_indices = r.choice(no_of_int_points,int(percent*no_of_int_points), replace=False)

    for jj in np.arange(no_of_int_points):

        if jj in sample_indices:

            hs_params_list[jj]["myofilament_parameters"][k][0] == base_value*20

    return hs_params_list

def df_fibrosis_law(dolfin_functions,base_value,k,percent,no_of_int_points):

    sample_indices = r.choice(no_of_int_points,int(percent*no_of_int_points), replace=False)
    #print "sample indices"
    #print sample_indices

    for jj in np.arange(no_of_int_points):

        if jj in sample_indices:

            if k == "cb_number_density":
                dolfin_functions[k][-1].vector()[jj] = base_value*20 #make 20 specified by user
            else:
                dolfin_functions["passive_params"][k][-1].vector()[jj] = base_value*20

    return dolfin_functions

def scalar_fiber_w_compliance_law(hs_params_list,base_value,k,fiber_value,no_of_int_points,geo_options):

    end_marker_array = geo_options["end_marker_array"]

    for jj in np.arange(no_of_int_points):

        if end_marker_array[jj] > 9.0 or end_marker_array[jj] < 1.0:
            hs_params_list[jj]["myofilament_parameters"][k][0] = fiber_value

    return hs_params_list

def df_fiber_w_compliance_law(dolfin_functions,base_value,k,fiber_value,no_of_int_points,geo_options):

    end_marker_array = geo_options["end_marker_array"]

    for jj in np.arange(no_of_int_points):

        if end_marker_array[jj] > 9.0 or end_marker_array[jj] < 1.0:
            if k == "cb_number_density":
                dolfin_functions[k][-1].vector()[jj] = fiber_value
            else:
                dolfin_functions["passive_params"][k][-1].vector()[jj] = fiber_value

    return dolfin_functions
