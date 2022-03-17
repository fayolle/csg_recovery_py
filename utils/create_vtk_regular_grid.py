# - Create a c++ file corresponding to the expression generated by the GP
# - Compile everything
# - Run the compiled binary and output a vtk file


import sys
import os
import math

import create_eval_source
import point_set


def compute_vector_length(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def main(fit_filename, exp_filename, cpp_filename, ps_filename, vtk_filename):
    create_eval_source.main(fit_filename, exp_filename, 
                            cpp_filename, ps_filename)
    
    # keep the part before the '.'
    basename = cpp_filename.split('.')[0]
    # keep only the name after the last '/'
    basename = basename.split('/')[-1]
    # program name := cpp_filename - .cpp
    prog_name = basename

    # generate the command line for compiling the program
    compile_line = 'g++ operations.cpp primitives.cpp '
    compile_line = compile_line + 'sample_expression_grid.cpp '
    compile_line = compile_line + cpp_filename
    compile_line = compile_line + ' -o ' + prog_name
    print(compile_line)

    # check for compilation error
    os.system(compile_line)
    
    # get the bounding box of the point-set
    ps = point_set.read_point_set(ps_filename)
    bbox = point_set.compute_bounding_box(ps)

    # enlarge the bounding box
    scale = 0.1
    p_min, p_max = bbox
    x_min, y_min, z_min = p_min
    x_max, y_max, z_max = p_max
    diag = (x_max-x_min, y_max-y_min, z_max-z_min)
    bbox_diag_length = compute_vector_length(diag)
    new_p_min = (x_min - scale*bbox_diag_length, 
                 y_min - scale*bbox_diag_length, 
                 z_min - scale*bbox_diag_length)
    new_p_max = (x_max + scale*bbox_diag_length, 
                 y_max + scale*bbox_diag_length, 
                 z_max + scale*bbox_diag_length)
    larger_bbox = (new_p_min, new_p_max)
    
    # get the number of subdivisions (arguments to the program?)
    #grid_subdivisions = (25,25,25)
    grid_subdivisions = (128,128,128)

    # run
    # create the command line
    create_vtk = './' + prog_name + ' '  
    # the bounding box information
    create_vtk = create_vtk + str(new_p_min[0]) + ' ' 
    create_vtk = create_vtk + str(new_p_min[1]) + ' ' 
    create_vtk = create_vtk + str(new_p_min[2]) + ' '
    create_vtk = create_vtk + str(new_p_max[0]) + ' ' 
    create_vtk = create_vtk + str(new_p_max[1]) + ' '
    create_vtk = create_vtk + str(new_p_max[2]) + ' '
    # the subdivision information
    create_vtk = create_vtk + str(grid_subdivisions[0]) + ' '
    create_vtk = create_vtk + str(grid_subdivisions[1]) + ' '
    create_vtk = create_vtk + str(grid_subdivisions[2]) + ' '
    # the name of the vtk file
    create_vtk = create_vtk + vtk_filename
    # run
    print(create_vtk)
    os.system(create_vtk)


def usage(prog):
    print('Usage: ')
    print(prog + ' ' + 'ex.fit ex.txt ex.cpp ex.xyzn ex.vtk')
    print('where:')
    print('\tex.fit: a list of fitted primitives for ex.xyzn')
    print('\tex.txt: a model for ex.xyzn')
    print('\tex.cpp: the model ex.txt translated to c++')
    print('\tex.xyzn: the finite point-set')
    print('\tex.vtk: the generated vtk grid with samples of the model')


if __name__ == '__main__':
    num_args = len(sys.argv)
    if num_args != 6:
        usage(sys.argv[0])
        sys.exit(1)

    fit_filename = sys.argv[1]
    exp_filename = sys.argv[2]
    cpp_filename = sys.argv[3]
    ps_filename = sys.argv[4]
    vtk_filename = sys.argv[5]
    
    main(fit_filename, exp_filename, cpp_filename, ps_filename, vtk_filename)
