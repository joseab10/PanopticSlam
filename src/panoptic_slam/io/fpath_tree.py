from os import path
import re


class FPathTree:
    """
        Class for defining a directory tree structure, allowing for formatted variables.
    """

    def __init__(self, name, desc=None, children=None):

        self._args = None
        self._children = None
        self._up_path = None
        self._parent = None
        self._path = None
        self.desc = desc
        self.name = name
        self.children = children

    # Properties
    # ----------------------------------------
    @property
    def name(self):
        """
        Getter for the Name property.

        :return: (str) Name of the node (without variable replacement).
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        Setter for the Name property. If it contains variables, they must each have a name and
        use Python's formatting conventions, e.g.: {var:05d}.

        :param value: (str) Name of the node.

        :return: None
        """
        self._name = value
        self._build_args_list()

    @property
    def parent(self):
        """
        Getter for the Parent property.
        READ-ONLY:
        Parents are set when adding children to avoid infinite recursion.

        :return: (FPathTree) Parent of this node.
        """
        return self._parent

    @property
    def children(self):
        """
        Getter for the Children property.

        :return: (list[FPathTree]) List of children of this node.
        """
        if self._children is None:
            self._children = []

        return self._children

    @children.setter
    def children(self, value):
        """
        Setter for the Children property.
        If node already had children, they are removed from the child list and become orphans.

        :param: (list[FPathTree]) List of children to be set to this node.

        :return: None
        """
        if self._children:
            for child in self._children:
                self.remove_child(child)

        if not value:
            self._children = []
            return

        try:
            for child in value:
                self.add_child(child)
        except TypeError:
            raise TypeError("Invalid type for children. It must be an iterable containing individual children objects")

    @property
    def path(self):
        """
        Getter for the Path property.
        READ-ONLY
        Path is computed upwards automatically when a child is added.

        :return: (list[FPathTree]) List of nodes starting from a root and going down the tree until this node.
        """
        if self._path is None:
            self._build_path_list()

        return self._path

    @property
    def args(self):
        """
        Getter for the Args property.
        READ-ONLY
        Formatting args are computed automatically when setting the Name.

        :return: (list[str]) List of variable names required for formatting the Name property.
        """
        if self._args is None:
            self._build_args_list()
        return self._args

    @property
    def path_args(self):
        """
        Getter for the Args of the full path property.
        READ-ONLY
        Formatting args are computed automatically when setting the Name and
        Path is computed upwards automatically when a child is added.

        :return: (list[str]) List of variable names required for formatting the fpath method.
        """
        args = set([])
        for p in self._path:
            args.update(p.args)

        return list(args)

    @property
    def tree_str(self):
        return self.str_representation()

    # Private (internal) building methods
    # ----------------------------------------
    def _build_args_list(self):
        """
        Builds the argument list from the name and stores it in the self._args private field.
        It searches for variable names enclosed within curly brackets "{}",
        discarding any formatting information starting with a colon ":".

        :return: None
        """
        if self._name is None:
            self._args = []
            return

        # Get all variables to be formatted (between curly braces {}, but ignoring formatting info after :)
        args_list = re.findall(r"{(\w+)(?::[a-zA-Z0-9]*)?}", self._name)
        # Discard repeated variable names
        args_list = set(args_list)
        self._args = list(args_list)

    def _build_child_path_list(self, parent_path=None):
        for child in self.children:
            if isinstance(child, FPathTree):
                child._build_path_list(parent_path=parent_path)

    def _build_path_list(self, parent_path=None):
        """
        Builds the Path as a list of nodes starting from a root and going down the tree until this node and stores it
        in the self._path field.
        Duplicate detection is used to avoid infinite loops, so whenever the same node is found a second time, it stops.

        :return: None
        """
        node = self

        tmp_path = []
        if parent_path is not None:
            tmp_path = [node for node in parent_path]

            if self in tmp_path:
                tmp_path.append(self)
                return

            tmp_path.append(self)
            self._path = tmp_path
            self._build_child_path_list(self._path)
            return

        while node is not None:
            tmp_path.append(node)
            node = node.parent

            # Avoid infinite loops with duplicate detection
            if node in tmp_path:
                break

        # Path is built from the bottom up, reverse it to get it from the top down.
        tmp_path.reverse()
        self._path = tmp_path
        self._build_child_path_list(self._path)

    # Other methods
    # ----------------------------------------
    def reset_parent(self):
        """
        Sets the parent to None, effectively orphaning ourselves.

        :return: None
        """
        self._parent = None

    def add_child(self, child):
        """
        Appends a single child to our existing children list.
        It also sets the child's parent to be this node, and recompute the child's path.

        :param child: (FPathTree) Node to be added as child of this node.

        :return: None
        """
        if not isinstance(child, FPathTree):
            raise TypeError("Invalid type for a child node. Only PathTree objects currently supported.")

        if self._children is None:
            self._children = []

        child._parent = self
        self._children.append(child)
        child._build_path_list(self.path)

    def remove_child(self, child):
        """
        Removes a single child to our existing children list.
        It also resets the child's parent to be None, and recompute the child's path.

        :param child: (FPathTree) Node to be removed from this node's child list.

        :return: None
        """
        if not isinstance(child, FPathTree):
            raise TypeError("Invalid type for a child node. Only PathTree objects currently supported.")

        child.reset_parent()
        self._children.remove(child)
        child._build_path_list()

    def format(self, **kwargs):
        """
        Formats the name of the node using the passed variables.
        If the name is deemed to not contain any variables, then it is returned as is.

        :param kwargs:  Key-Value for the variables to be replaced.
                        Mandatory Keys can be obtained from the self.args property

        :return: (str) Name of the node with variables replaced and formatted.
        """
        if not self._args:
            return self._name

        str_fmt_args = {arg: kwargs[arg] for arg in self._args}
        return self._name.format(**str_fmt_args)

    def fpath(self, **kwargs):
        """
        Provides the Path of the node as a string with all of the variables replaced and formatted.

        :param kwargs:  Key-Value for the variables to be replaced.
                        Mandatory Keys can be obtained from the self.args property

        :return: (str) Path of the node with variables replaced and formatted
        """
        path_list = [d.format(**kwargs) for d in self.path]
        # Remove empty strings
        path_list = list(filter(None, path_list))
        # Return a path
        tmp_path = path.join(*path_list)
        tmp_path = path.expanduser(tmp_path)
        tmp_path = path.expandvars(tmp_path)
        return tmp_path

    def str_representation(
            self, multiline=True, indent="  ", indent_level=0, bullet="+ ", delim=None, show_desc=True,
            traversed_nodes=None):
        """
        Generates a string representation of the full tree starting from this node and recursively traversing the tree
        downwards.

        :param multiline: (bool, Default: True) String will use line breaks. Better for readability.
        :param indent: (str, Default: "  ") String to be used for indenting the different levels of the tree.
        :param indent_level: (int, Default: 0) Indentation level of the current level. Indentation is achieved by
                                repeating indent_level times the indent string.
        :param bullet: (str, Default: "+ ") Bullet to be added before each node in the tree.
        :param delim: (list[str], Default: None) Delimiters to be used for grouping child nodes.
                        If None, then ["{", "}"] will be used.
        :param show_desc: (bool, Default: True) Shows the node's description inside <> next to the name, if it exists.
        :param traversed_nodes: (dict{FPathTree:bool}, Default: None) List of traversed nodes,
                                to avoid duplicity and infinite recursion.

        :return: (str) String representing the full tree with this node as its root.
        """
        indentation = indent * indent_level if indent_level else ""
        node_prefix = indentation + bullet
        repr_str = node_prefix + self._name

        if show_desc and self.desc is not None:
            repr_str += " <" + self.desc + ">"

        if delim is None:
            delim = [": {", "}"]

        dl, dr = delim

        lb = " "
        if multiline:
            lb = "\n"

        if traversed_nodes is None:
            traversed_nodes = {}

        if self in traversed_nodes:
            repr_str += dl + lb + indentation + indent + " ... (Repeated Node)" + lb + indentation + dr
            return repr_str

        traversed_nodes[self] = True

        if self._children:
            repr_str += dl + lb + ("," + lb).join([
                child.str_representation(
                    multiline=multiline, indent=indent, indent_level=indent_level + 1, bullet=bullet,
                    delim=delim, show_desc=show_desc, traversed_nodes=traversed_nodes
                )
                for child in self._children]) + lb + indentation + dr

        return repr_str

    def __str__(self):
        """
        :return: (str) String representation of the Tree with this node as its root.
        """
        rep = self._name
        if self.desc:
            rep += "<" + self.desc + ">"
        return rep

    def __repr__(self):
        """
        :return: (str) String representation of the Tree with this node as its root.
        """
        return str(self)

    def __len__(self):
        """
        :return: (int) Number of children of this node.
        """
        return len(self._children)

    def __getitem__(self, item):
        """
        :param: (int) Index of the child to be returned.

        :return: (FPathTree) Child with index <item>
        """
        return self._children[item]
