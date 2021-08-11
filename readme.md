# Segmenter-based-on-OpenMMLab
"Segmenter: Transformer for Semantic Segmentation, arxiv 2105.05633." reproduced via mmsegmentation.

We reproduce [Segmenter](https://arxiv.org/pdf/2105.05633.pdf) via [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 
based on [official open-sourced code](https://github.com/rstrudel/segmenter).

## Environment
- python=3.7

- pytorch=1.7.1

- torchvision=0.8.2

- cudatoolkit=10.1

- mmcv-full=1.3.10

- mmsegmentation=0.16.0

**Note:** You should install `pytorch` with a version higher than `1.7`, because the pretrained model of `Deit` is saved via 1.7+ 
`pytorch`. Otherwise you may encounter some errors while loading the `state_dict`.

## Results on ADE20K

The passwds of download links are all 'nopw'.
<table>
  <tr>
    <th>Exp</th>
    <th>Name</th>
    <th>backbone</th>
    <th>Our mIoU-SS </th>
    <th>mIoU in paper</th>
    <th>Resolution</th>
    <th>BS</th>
    <th colspan="3">Download</th>
  </tr>
<tr>
    <td>4th line in Table3</td>
    <td>Seg-B<span>&#8224;</span>-Linear/16</td>
    <td>Deit</td>
    <td> 46.69 </td>
    <td> 47.10 </td>
    <td>512x512</td>
    <td>8</td>
    <td><a href="https://pan.baidu.com/s/1M5oNIAjeiKjM22FydEUlMg">model</a></td>
    <td><a href="https://drive.google.com/file/d/1Mt3vxst6_fBi_iaJxfXKOh4bO2fI4b9r/view?usp=sharing">config</a></td>    
    <td><a href="https://drive.google.com/file/d/1wLjCIyG8gP5OUC0vImRVmOtxdm6kmZ-W/view?usp=sharing">log</a></td>
</tr>
<tr>
    <td>4th line in Table6</td>
    <td>Seg-B<span>&#8224;</span>-Mask/16</td>
    <td>Deit</td>
    <td> 48.41 </td>
    <td> 47.67 </td>
    <td>512x512</td>
    <td>8</td>
    <td><a href="https://pan.baidu.com/s/1Rs6xDy2R5YAu5cSsPd2abA">model</a></td>
    <td><a href="https://drive.google.com/file/d/1SDWrkzjvZiwRQLDdgdvEAL30qYsOAbYP/view?usp=sharing">config</a></td>
    <td><a href="https://drive.google.com/file/d/1iPSTNzMsumptvfCUBd2X4yUDNc-yFqM1/view?usp=sharing">log</a></td>
</tr>
</table>