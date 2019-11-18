class Reporter:
  @staticmethod
  def write_output(
    file_name = "output.txt",
    values = []
  ):
    f = open(file_name, "w+")
    
    content = ','.join(map(str, values))

    f.write(
      ""
      + content
    )

    f.close()
